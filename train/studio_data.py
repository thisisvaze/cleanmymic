"""Self-contained permissive studio-restoration data pipeline for StudioNet.

Key differences from CleanMyMic's earlier suppressor pipeline:
1. Time-aligned DRY ANECHOIC targets (no 50 ms early reflections — matches RE-USE / StuPASE 2026).
2. Full-band 48 kHz targets (input band-limiting, mic EQ, clipping, codec artifacts, and dropouts NEVER hit the target).
3. Strictly permissive / share-alike training pools (CC0 / CC BY / CC BY-SA / Apache 2.0) so trained weights ship MIT.
4. Measured microphone impulse responses (Surrey Mic IRs, CC BY 4.0) + synthetic Bluetooth/VoIP/suppressor artifacts.
5. Real recorded room pairs (RealMAN ma_speech/dp_speech, CC BY 4.0) mixed into training batches.
"""

import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

from model import SR

LOW_CUT = 70  # Hz: neutral studio low cut
EVENT_LEN = 4  # s: clips <= 4 s used as transient events

# Strictly permissive / share-alike 48 kHz clean speech pools (no noncommercial EARS/Expresso)
STUDIO_CLEAN = {
    "packed5/src/vctk1": 0.45,  # 41.6 h, 48 kHz DPA studio mic (CC BY 4.0)
    "packed5/src/slr83": 0.30,  # 31.3 h, 48 kHz UK/Irish accents, -79 dB floor (CC BY-SA 4.0)
    "packed5/src/aishell3": 0.20,  # 85.6 h, 44.1->48 kHz studio Mandarin (Apache 2.0)
    "packed5/src/vocalset": 0.05,  # 8.8 h, 44.1->48 kHz vocals/breaths (CC BY 4.0)
}
STUDIO_CLEAN_POLISH = {
    "packed5/src/vctk1": 0.90,
    "packed5/src/vocalset": 0.10,
}
STUDIO_DEHISS = {
    "packed5/src/vctk1",
    "packed5/src/slr83",
    "packed5/src/aishell3",
    "packed5/src/vocalset",
}

# Strictly permissive / share-alike noise beds & events (427+ h already packed)
STUDIO_BEDS = {
    "packed5/src/realman": 0.20,  # 57.3 h real cafes, malls, streets, offices (CC BY 4.0)
    "packed5/src/cochlscene": 0.18,  # 65.6 h cafes, restaurants, kitchens (CC BY-SA 3.0)
    "packed6/src/cochl_indoor": 0.12,  # 49.0 h indoor scenes (CC BY-SA 3.0)
    "packed6/src/cochl_outdoor": 0.10,  # 47.9 h outdoor scenes (CC BY-SA 3.0)
    "packed6/src/cochl_transit": 0.10,  # 48.9 h transit scenes (CC BY-SA 3.0)
    "packed5/src/fsd50k": 0.12,  # 79.8 h CC0/CC BY filtered non-vocal
    "packed5/src/sonyc": 0.09,  # 51.4 h urban streets (CC BY 4.0)
    "packed5/src/dns": 0.07,  # 27.0 h DNS Freesound CC0 (breath/spit removed)
    "packed5/src/demand": 0.02,  # 0.8 h real room/cafe ambiences (CC BY-SA 3.0)
}
STUDIO_EVENTS = {
    "packed5/src/fsd50k": 0.65,
    "packed5/src/dns": 0.35,
}


class Pool:
    """Packed int16 audio pool (memmap-friendly) matching packed5/packed6 format."""

    def __init__(
        self, path, device, dehissed=False, memmap=True, split=None, max_len=None
    ):
        mode = "c" if memmap else None
        self.dev, store = device, "cpu" if memmap else device
        self.xs, src, start, length, voice, weight = [], [], [], [], [], []
        for k, (p, share) in enumerate(
            (path if isinstance(path, dict) else {path: 1.0}).items()
        ):
            p_str = str(p)
            use_dh = dehissed is True or (dehissed and p in dehissed)
            npy_path = (
                f"{p_str}_dehissed.npy"
                if use_dh and Path(f"{p_str}_dehissed.npy").exists()
                else f"{p_str}.npy"
            )
            x, idx = np.load(npy_path, mmap_mode=mode), np.load(f"{p_str}_idx.npy")
            lens = idx[1:] - idx[:-1]
            keep = np.ones(len(lens), bool)
            if split:
                meta = np.load(f"{p_str}_meta.npz")
                keep &= meta["val"] == (split == "val")
                v = meta["voice"]
            else:
                v = np.zeros(len(lens), bool)
            if max_len:
                keep &= lens <= max_len * SR
            self.xs.append(torch.from_numpy(x).to(store))
            src.append(np.full(keep.sum(), k))
            start.append(idx[:-1][keep])
            length.append(lens[keep])
            voice.append(v[keep])
            weight.append(lens[keep] / max(lens[keep].sum(), 1) * share)
        self.src, self.start, self.len, self.voice = (
            torch.from_numpy(np.concatenate(a)).to(store)
            for a in (src, start, length, voice)
        )
        self.w = torch.from_numpy(np.concatenate(weight)).float()

    def sample(self, n, seg, wrap=False, flags=False):
        c = torch.multinomial(self.w, n, replacement=True)
        off = (
            torch.rand(n, device=self.start.device) * (self.len[c] - seg).clamp(min=0)
        ).long()
        out = torch.zeros(n, seg, dtype=torch.int16, device=self.start.device)
        for i, (k, s, l, o) in enumerate(
            zip(
                self.src[c].tolist(),
                self.start[c].tolist(),
                self.len[c].tolist(),
                off.tolist(),
            )
        ):
            clip = self.xs[k][s + o : s + l]
            if wrap and l < seg:
                clip = clip.repeat(-(-seg // l))
            out[i, : min(seg, len(clip))] = clip[:seg]
        out = out.to(self.dev, non_blocking=True).float() / 32768.0
        return (out, self.voice[c].to(self.dev)) if flags else out

    def joined(self, n, seg, slots=4, gap=(0.1, 1.5), hold=None):
        slots = max(slots, seg // SR)
        out = torch.zeros(n, seg, dtype=torch.int16, device=self.start.device)
        at = (-torch.rand(n) * self.len.float().mean().item()).long().tolist()
        for _ in range(slots):
            c = torch.multinomial(self.w, n, replacement=True)
            gaps = (torch.empty(n).uniform_(*gap) * SR).long().tolist()
            for i, (k, s, l) in enumerate(
                zip(self.src[c].tolist(), self.start[c].tolist(), self.len[c].tolist())
            ):
                lo, hi = max(at[i], 0), min(at[i] + l, seg)
                if lo < hi:
                    out[i, lo:hi] = self.xs[k][s + lo - at[i] : s + hi - at[i]]
                at[i] += l + gaps[i]
                if hold and at[i] > hold[0][i]:
                    at[i] += hold[1][i]
                    hold[1][i] = 0
        return out.to(self.dev, non_blocking=True).float() / 32768.0


def random_eq_db(n, f, knots=10, std_db=4.5):
    pos = (
        (f.clamp(min=50).log() - math.log(50)) / math.log(24000 / 50) * (knots - 1)
    ).clamp(max=knots - 1)
    i0 = pos.floor().long().clamp(max=knots - 2)
    w = pos - i0
    g = torch.randn(n, knots, device=f.device) * std_db
    return g[:, i0] * (1 - w) + g[:, i0 + 1] * w


def highpass(f, fc=LOW_CUT):
    s, w = 2j * math.pi * f, 2 * math.pi * fc
    return s**2 / (s**2 + math.sqrt(2) * w * s + w**2)


def align_rirs(rirs):
    """Shift every RIR so its direct-path peak sits at t=0 (time-shifted anechoic target, RE-USE 2026 §2.1).

    Ensures both synthetic and measured real-room RIR banks (BUT, ACE, Arni) stay time-aligned to dry targets.
    """
    peaks = rirs.abs().argmax(dim=1)
    if (peaks == 0).all():
        return rirs
    out = torch.zeros_like(rirs)
    for i, p in enumerate(peaks.tolist()):
        rem = rirs.shape[1] - p
        if rem > 0:
            out[i, :rem] = rirs[i, p:]
    return out


def by_distance(rirs):
    drr = rirs[:, : SR // 400].pow(2).sum(1) / rirs[:, SR // 400 :].pow(2).sum(1).clamp(
        min=1e-12
    )
    order = drr.argsort(descending=True)
    return order[: len(order) * 3 // 5], order[len(order) * 3 // 5 :]


def active_rms(a, frame=SR // 50):
    e = a[:, : a.shape[1] // frame * frame].unflatten(1, (-1, frame)).pow(2).mean(2)
    act = e >= e.amax(1, keepdim=True) * 1e-4
    return (
        ((e * act).sum(1, keepdim=True) / act.sum(1, keepdim=True).clamp(min=1))
        .sqrt()
        .clamp(min=1e-5)
    )


def device_noise(n, L, f, level, dev):
    rand = lambda lo, hi: torch.empty(n, 1, device=dev).uniform_(lo, hi)
    coin = lambda p: torch.rand(n, 1, device=dev) < p
    white = lambda: torch.fft.rfft(torch.randn(n, L, device=dev))

    def at_level(spec, db):
        return (
            spec
            / (spec.abs().pow(2).sum(1, keepdim=True) * 2 / L**2)
            .sqrt()
            .clamp(min=1e-12)
            * level
            * 10 ** (db / 20)
        )

    fr = f.clamp(min=1)
    slope = rand(4, 9) / 6.02
    shape = (
        (fr / 40) ** -slope / (1 + (30 / fr) ** 4) / (1 + (fr / rand(1000, 4000)) ** 4)
    )
    rumble = at_level(white() * shape, -rand(10, 35)) * coin(0.5)
    t, tone_f = torch.arange(L, device=dev) / SR, rand(40, 400)
    tones = sum(
        torch.sin(2 * math.pi * (k * tone_f * t + torch.rand(n, 1, device=dev))) / k
        for k in (1, 2, 3)
    )
    rumble = rumble + at_level(torch.fft.rfft(tones), -rand(20, 40)) * coin(0.15)
    shape = (fr / 4000) ** (rand(-3, 3) / 6.02) / (1 + (2000 / fr) ** 4)
    hiss = at_level(white() * shape, -rand(40, 75)) * coin(0.6)
    return rumble, hiss


def load_mic_irs(path, ir_len=2048, device="cpu"):
    """Load pre-packed float32 mic IR matrix [M, ir_len] if present."""
    p = Path(path)
    if not p.exists():
        return None
    irs = np.load(p).astype(np.float32)
    if irs.ndim != 2 or irs.shape[0] == 0:
        return None
    if irs.shape[1] > ir_len:
        irs = irs[:, :ir_len]
    elif irs.shape[1] < ir_len:
        irs = np.pad(irs, ((0, 0), (0, ir_len - irs.shape[1])))
    return torch.from_numpy(irs).to(device)


class RealPairPool:
    """Real recorded room/device pairs (RealMAN <room>/mic/<id>_CH0.flac <-> <room>/ref/<id>.flac, CC BY 4.0)."""

    def __init__(self, pairs_dir, device="cpu", max_clips=1200):
        self.dev = device
        self.loaded = []
        root = Path(pairs_dir)
        if not root.exists():
            return
        for mic_path in sorted(root.rglob("mic/*_CH0.flac")):
            stem = mic_path.stem.removesuffix("_CH0")
            ref_path = mic_path.parent.parent / "ref" / f"{stem}.flac"
            if not ref_path.exists():
                continue
            x, sr1 = sf.read(mic_path, dtype="float32")
            t, sr2 = sf.read(ref_path, dtype="float32")
            if x.ndim > 1:
                x = x[:, 0]
            if t.ndim > 1:
                t = t[:, 0]
            if sr1 != SR:
                x = resample_poly(x, SR, sr1).astype(np.float32)
            if sr2 != SR:
                t = resample_poly(t, SR, sr2).astype(np.float32)
            n = min(len(x), len(t))
            if n < SR:
                continue
            self.loaded.append(
                (torch.from_numpy(x[:n].copy()), torch.from_numpy(t[:n].copy()))
            )
            if len(self.loaded) >= max_clips:
                break

    def __len__(self):
        return len(self.loaded)

    def sample(self, n, seg):
        xs = torch.zeros(n, seg)
        ts = torch.zeros(n, seg)
        for i in range(n):
            idx = int(torch.randint(len(self.loaded), ()))
            x, t = self.loaded[idx]
            if len(x) >= seg:
                off = int(torch.randint(len(x) - seg + 1, ()))
                xs[i], ts[i] = x[off : off + seg], t[off : off + seg]
            else:
                xs[i, : len(x)], ts[i, : len(t)] = x, t
        return xs.to(self.dev, non_blocking=True), ts.to(self.dev, non_blocking=True)


class TeacherPairPool:
    """Verified offline teacher pseudo-pairs (`noisy/<id>.flac` <-> `clean/<id>.flac`, e.g. NemotronLabs RE-USE)."""

    def __init__(self, pairs_dir, device="cpu", max_clips=4000):
        self.dev = device
        self.loaded = []
        root = Path(pairs_dir)
        noisy_dir, clean_dir = root / "noisy", root / "clean"
        if not (noisy_dir.exists() and clean_dir.exists()):
            return
        for noisy_path in sorted(noisy_dir.iterdir()):
            if noisy_path.suffix.lower() not in (".flac", ".wav"):
                continue
            clean_path = clean_dir / noisy_path.name
            if not clean_path.exists():
                clean_path = clean_dir / f"{noisy_path.stem}.flac"
            if not clean_path.exists():
                clean_path = clean_dir / f"{noisy_path.stem}.wav"
            if not clean_path.exists():
                continue
            x, sr1 = sf.read(noisy_path, dtype="float32")
            t, sr2 = sf.read(clean_path, dtype="float32")
            if x.ndim > 1:
                x = x[:, 0]
            if t.ndim > 1:
                t = t[:, 0]
            if sr1 != SR:
                x = resample_poly(x, SR, sr1).astype(np.float32)
            if sr2 != SR:
                t = resample_poly(t, SR, sr2).astype(np.float32)
            n = min(len(x), len(t))
            if n < SR:
                continue
            self.loaded.append(
                (torch.from_numpy(x[:n].copy()), torch.from_numpy(t[:n].copy()))
            )
            if len(self.loaded) >= max_clips:
                break

    def __len__(self):
        return len(self.loaded)

    def sample(self, n, seg):
        xs = torch.zeros(n, seg)
        ts = torch.zeros(n, seg)
        for i in range(n):
            idx = int(torch.randint(len(self.loaded), ()))
            x, t = self.loaded[idx]
            if len(x) >= seg:
                off = int(torch.randint(len(x) - seg + 1, ()))
                xs[i], ts[i] = x[off : off + seg], t[off : off + seg]
            else:
                xs[i, : len(x)], ts[i, : len(t)] = x, t
        return xs.to(self.dev, non_blocking=True), ts.to(self.dev, non_blocking=True)


def _apply_spectral_artifacts(noisy, dev):
    """Simulate lossy Bluetooth HFP/LC3/VoIP codecs, PLC repeats, micro-dropouts, and upstream noise-suppressors."""
    n, seg = noisy.shape
    coin = lambda p: torch.rand(n, 1, device=dev) < p

    # 1) Short packet-loss / Bluetooth micro-dropouts & PLC frame repeats (10-40 ms, 14% of clips)
    has_drop = coin(0.14)[:, 0]
    if has_drop.any():
        noisy = noisy.clone()
        for i in torch.nonzero(has_drop, as_tuple=False)[:, 0].tolist():
            for _ in range(int(torch.randint(1, 3, ()))):
                dlen = int(torch.randint(SR // 133, SR // 25, ()))  # 7.5 - 40 ms
                if dlen < seg:
                    pos = int(torch.randint(0, seg - dlen, ()))
                    if torch.rand(()) < 0.5 and pos >= dlen:
                        # Bluetooth HFP/LC3 PLC: repeat prior packet with slight attenuation
                        noisy[i, pos : pos + dlen] = noisy[i, pos - dlen : pos] * 0.85
                    else:
                        ramp = min(96, dlen // 4)
                        win = torch.zeros(dlen, device=dev)
                        if ramp > 0:
                            r = torch.linspace(1.0, 0.0, ramp, device=dev)
                            win[:ramp] = r
                            win[-ramp:] = r.flip(0)
                        noisy[i, pos : pos + dlen] *= win

    # 2) Bluetooth mSBC / LC3 / Opus subband quantization + upstream suppressor gating holes (24% of clips)
    has_art = coin(0.24)[:, 0]
    if has_art.any():
        win = torch.hann_window(960, device=dev)
        sub = noisy[has_art]
        S = torch.stft(sub, 960, 480, window=win, return_complex=True)
        mag = S.abs()
        m, nf, nt = sub.shape[0], S.shape[1], S.shape[2]
        grid = torch.rand(m, 1, max(1, nf // 12), max(1, nt // 4), device=dev)
        holes = F.interpolate((grid > 0.12).float(), size=(nf, nt), mode="nearest")[
            :, 0
        ]
        holes[:, :24, :] = 1.0

        # Mode A: coarse codec subband magnitude quantization + zeroed quiet bins (mSBC/LC3/low-bitrate Opus)
        peak_m = mag.amax((1, 2), keepdim=True).clamp(min=1e-6)
        step = peak_m * torch.empty(m, 1, 1, device=dev).uniform_(0.008, 0.025)
        q_mag = torch.round(mag / step) * step
        phase = S / (mag + 1e-8)
        S_codec = q_mag * phase
        # Mode B: upstream noise suppressor (Voice Isolation / Zoom DSP): steep >10-11.5 kHz rolloff + consonant gating
        freqs = torch.fft.rfftfreq(960, 1.0 / SR, device=dev)[None, :, None]
        sup_cut = torch.empty(m, 1, 1, device=dev).uniform_(9500.0, 11800.0)
        sup_lp = torch.sigmoid((sup_cut - freqs) / 250.0)
        sup_gate = torch.where(mag < peak_m * 0.025, 0.12, 1.0)
        jitter = torch.randn(m, nf, nt, device=dev) * 0.25 * (freqs > 3500.0)
        S_sup = (
            S * sup_lp * sup_gate * torch.complex(torch.cos(jitter), torch.sin(jitter))
        )

        mode_pick = torch.rand(m, 1, 1, device=dev)
        S_mod = torch.where(
            mode_pick < 0.40,
            S_codec,
            torch.where(mode_pick < 0.72, S_sup, S * (0.2 + 0.8 * holes)),
        )
        noisy = noisy.clone()
        noisy[has_art] = torch.istft(S_mod, 960, 480, window=win, length=seg)

    return noisy


def _wind_gusts(n, seg, level, dev):
    """Procedural outdoor wind-on-mic buffeting: slow modulated 20-450 Hz gusts (walking/sitting along roads)."""
    w = torch.randn(n, seg, device=dev)
    f = torch.fft.rfftfreq(seg, 1 / SR, device=dev)
    # Steep low-pass around 180-450 Hz with 1/f^1.5 rolloff (classic capsule turbulence)
    fc = torch.empty(n, 1, device=dev).uniform_(180.0, 450.0)
    shape = (1.0 / (1.0 + (f / fc) ** 3)) * (
        1.0 / (1.0 + (22.0 / f.clamp(min=1.0)) ** 4)
    )
    gust = torch.fft.irfft(torch.fft.rfft(w) * shape, n=seg)
    # Slow random amplitude modulation (0.5-3 Hz gust bursts)
    m_pts = max(4, seg // (SR // 4))
    env = torch.rand(n, 1, m_pts, device=dev).pow(
        2.0
    )  # spiky bursts with quieter troughs
    env = F.interpolate(env, size=seg, mode="linear", align_corners=False)[:, 0]
    gust = gust * env
    g_rms = gust.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-6)
    snr = torch.empty(n, 1, device=dev).uniform_(-5.0, 18.0)
    return gust / g_rms * level * 10 ** (-snr / 20.0)


def make_studio_batch(
    clean,
    noise,
    rirs,
    n,
    seg,
    events=None,
    rooms=None,
    mic_irs=None,
    real_pairs=None,
    teacher_pairs=None,
    hold=0.0,
    clean_prob=0.15,
    bandlim_prob=0.45,
    noise_only_prob=0.0,
    wind_prob=0.18,
    force_bandlim_hz=None,
):
    """Build (noisy, target) 48 kHz studio-restoration pairs [n, seg]."""
    dev = rirs.device
    rand = lambda lo, hi: torch.empty(n, 1, device=dev).uniform_(lo, hi)
    coin = lambda p: torch.rand(n, 1, device=dev) < p
    rms = lambda a: a.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-5)

    quiet = None
    if hold:
        long = seg > 20 * SR
        quiet = torch.rand(n) < hold
        when = torch.empty(n).uniform_(*((3, 8) if long else (0.5, 3))) * SR
        spans = torch.empty(n).uniform_(*((8, 18) if long else (2, 5))) * SR * quiet
    speech = clean.joined(
        n, seg, hold=(when.long().tolist(), spans.long().tolist()) if hold else None
    )
    speech = speech / active_rms(speech) * 10 ** (rand(-48, -14) / 20)

    # Room reverb on INPUT only; target stays 100% TIME-ALIGNED DRY ANECHOIC (70 Hz studio HPF)
    # Use ALL rooms (including heavy conference/bedroom reverb!) when no competing talker is present,
    # and restrict primary talker to near rooms when a competing talker uses far rooms.
    L = seg + rirs.shape[1]
    f = torch.fft.rfftfreq(L, 1 / SR, device=dev)
    has_other = coin(0.45)
    if hold:
        has_other = has_other | quiet[:, None].to(dev)
    all_rirs = rirs[torch.randint(len(rirs), (n,), device=dev)]
    if rooms is not None:
        near_rirs = rirs[rooms[0][torch.randint(len(rooms[0]), (n,), device=dev)]]
        rir = torch.where(has_other, near_rirs, all_rirs)
    else:
        rir = all_rirs
    S = torch.fft.rfft(speech, n=L)
    in_room = coin(0.75)
    rev = torch.where(
        in_room, torch.fft.irfft(S * torch.fft.rfft(rir, n=L), n=L)[:, :seg], speech
    )
    dry_hp = torch.fft.irfft(S * highpass(f, LOW_CUT), n=L)[:, :seg]
    # Gently clean pure-silence room hum (< -48 dB below target speech, smoothed over 40 ms) so VCTK's -43 dB
    # studio rumble doesn't teach the model to leave noise haze in pauses; breaths/consonants stay untouched.
    fr_g = SR // 50  # 20 ms
    m_len = seg // fr_g * fr_g
    t_rms = dry_hp[:, :m_len].unflatten(1, (-1, fr_g)).pow(2).mean(2).sqrt()
    thr = active_rms(dry_hp) * 0.004  # -48 dB below active speech
    gate_f = (t_rms / (t_rms + thr)).pow(2)
    gate_f = F.avg_pool1d(gate_f[:, None], 5, 1, 2)[:, 0]  # smooth ~100 ms transitions
    gate_s = F.interpolate(
        gate_f[:, None], size=seg, mode="linear", align_corners=False
    )[:, 0]
    dry_clean = dry_hp * (0.01 + 0.99 * gate_s)
    target = dry_clean * (active_rms(rev) / active_rms(dry_clean))

    # Background noise (-8..35 dB unvoiced SNR covers loud balcony/cafe noise) + events + babble + competing talkers
    level = active_rms(rev)
    nz, voiced = noise.sample(n, seg, wrap=True, flags=True)
    snr = torch.where(voiced[:, None], rand(5, 35), rand(-8, 35))
    nz = nz / rms(nz) * level * 10 ** (-snr / 20) * coin(0.80)

    if events is not None:
        for k in range(3):
            ev, ev_voiced = events.sample(n, seg, flags=True)
            ev = ev.gather(
                1,
                (torch.arange(seg, device=dev) - torch.randint(seg, (n, 1), device=dev))
                % seg,
            )
            ev_snr = torch.where(ev_voiced[:, None], rand(5, 25), rand(-5, 25))
            nz = nz + ev / active_rms(ev) * level * 10 ** (-ev_snr / 20) * coin(
                0.55 - 0.18 * k
            )

    k_bab = 6
    voices = clean.sample(n * k_bab, seg).view(n, k_bab, seg) * (
        torch.rand(n, k_bab, 1, device=dev) < rand(0.3, 1.0)[..., None]
    )
    rk_bab = rirs[torch.randint(len(rirs), (n, k_bab), device=dev)]
    bab = torch.fft.irfft(
        (torch.fft.rfft(voices, n=L) * torch.fft.rfft(rk_bab, n=L)).sum(1), n=L
    )[:, :seg]
    nz = nz + bab / rms(bab) * level * 10 ** (-rand(10, 26) / 20) * coin(0.30)

    far_idx = rooms[1] if rooms is not None else torch.arange(len(rirs), device=dev)
    for k in range(2):
        other = clean.joined(n, seg)
        rk = rirs[far_idx[torch.randint(len(far_idx), (n,), device=dev)]]
        other_rev = torch.fft.irfft(
            torch.fft.rfft(other, n=L) * torch.fft.rfft(rk, n=L), n=L
        )[:, :seg]
        on = has_other if k == 0 else (has_other & coin(0.45))
        under = rand(5, 28)
        nz = nz + other_rev / active_rms(other_rev) * level * 10 ** (-under / 20) * on

    # Outdoor wind-on-mic buffeting gusts (walking/sitting along roads)
    if wind_prob > 0 and force_bandlim_hz is None:
        nz = nz + _wind_gusts(n, seg, level, dev) * coin(wind_prob)

    # Device noise floor + measured Surrey Mic IR + random mic coloration + Bluetooth/VoIP band-limiting (INPUT ONLY!)
    rumble, hiss = device_noise(n, L, f, level, dev)
    eq_db = random_eq_db(n, f, knots=10, std_db=4.5) * coin(0.75)
    mic = 10 ** (eq_db / 20)
    mic = mic * torch.where(coin(0.45), highpass(f, rand(100, 350)).abs(), 1.0)
    mic = mic * torch.where(coin(0.35), (1 + (f / rand(3500, 8000)) ** 2).rsqrt(), 1.0)

    if mic_irs is not None and len(mic_irs) > 0:
        mir = mic_irs[torch.randint(len(mic_irs), (n,), device=dev)]
        mir_mag = torch.fft.rfft(mir, n=L).abs()
        mid_mask = (f >= 300) & (f <= 3000)
        mir_mag = mir_mag / mir_mag[:, mid_mask].mean(1, keepdim=True).clamp(min=1e-4)
        mir_mag = mir_mag.clamp(0.05, 8.0)
        mic = mic * torch.where(coin(0.50), mir_mag, 1.0)

    if force_bandlim_hz is not None:
        cutoff = torch.full((n, 1), float(force_bandlim_hz), device=dev)
        lowpass = torch.sigmoid((cutoff - f) / 180.0)
    else:
        cuts = [
            4000.0,
            6000.0,
            7500.0,
            8000.0,
            8000.0,
            11025.0,
            12000.0,
            12000.0,
            16000.0,
        ]
        cutoff = torch.tensor(cuts, device=dev)[
            torch.randint(len(cuts), (n, 1), device=dev)
        ]
        lowpass = torch.where(
            coin(bandlim_prob), torch.sigmoid((cutoff - f) / 180.0), 1.0
        )

    noisy = torch.fft.irfft(
        ((torch.fft.rfft(rev + nz, n=L) + rumble) * mic + hiss) * lowpass, n=L
    )[:, :seg]

    # AGC pumping on input (normalized to unity mean gain during active speech so speech loudness stays matched!)
    fr = SR // 50
    e = noisy[:, : seg // fr * fr].unflatten(1, (-1, fr)).pow(2).mean(2)
    e = F.avg_pool1d(e[:, None], 25, 1, 12, count_include_pad=False)[:, 0]
    g = (level.pow(2) / e.clamp(min=1e-12)).pow(rand(0.2, 0.55) / 2).clamp(0.5, 8.0)
    sp_e = target[:, : seg // fr * fr].unflatten(1, (-1, fr)).pow(2).mean(2)
    sp_act = (sp_e >= sp_e.amax(1, keepdim=True) * 1e-3).float()
    g_sp = (g * sp_act).sum(1, keepdim=True) / sp_act.sum(1, keepdim=True).clamp(
        min=1.0
    )
    g = (g / g_sp.clamp(min=0.5)).clamp(0.5, 8.0)
    g = torch.where(
        coin(0.25), F.interpolate(g[:, None], size=seg, mode="linear")[:, 0], 1.0
    )
    noisy = noisy * g

    clip_lim = noisy.abs().amax(1, keepdim=True) * rand(0.35, 0.90)
    noisy = torch.where(coin(0.20), noisy.clamp(-clip_lim, clip_lim), noisy)
    if force_bandlim_hz is None:
        noisy = _apply_spectral_artifacts(noisy, dev)

    # Real recorded room pairs (RealMAN, ~10% of rows)
    real_end = 0
    if (
        real_pairs is not None
        and len(real_pairs) > 0
        and force_bandlim_hz is None
        and seg <= 12 * SR
    ):
        n_real = max(1, n // 10)
        rx, rt = real_pairs.sample(n_real, seg)
        sc = 10 ** (
            torch.empty(n_real, 1, device=dev).uniform_(-42, -16) / 20
        ) / active_rms(rt)
        rx, rt = rx * sc, rt * sc
        rt = torch.fft.irfft(
            torch.fft.rfft(rt, n=seg)
            * highpass(torch.fft.rfftfreq(seg, 1 / SR, device=dev), LOW_CUT),
            n=seg,
        )
        rnz = noise.sample(n_real, seg, wrap=True)
        rsnr = torch.empty(n_real, 1, device=dev).uniform_(5, 30)
        rx = rx + rnz / rms(rnz) * active_rms(rt) * 10 ** (-rsnr / 20) * (
            torch.rand(n_real, 1, device=dev) < 0.5
        )
        noisy[:n_real], target[:n_real] = rx, rt
        real_end = n_real

    # Verified offline teacher pseudo-pairs (e.g. NVIDIA-NemotronLabs-RE-USE OpenMDW-1.1, ~15% of rows)
    if (
        teacher_pairs is not None
        and len(teacher_pairs) > 0
        and force_bandlim_hz is None
        and seg <= 12 * SR
        and real_end < n
    ):
        n_tp = min(n - real_end, max(1, (n * 3) // 20))
        tx, tt = teacher_pairs.sample(n_tp, seg)
        sc = 10 ** (
            torch.empty(n_tp, 1, device=dev).uniform_(-42, -16) / 20
        ) / active_rms(tt)
        tx, tt = tx * sc, tt * sc
        f_seg = torch.fft.rfftfreq(seg, 1 / SR, device=dev)
        tt = torch.fft.irfft(
            torch.fft.rfft(tt, n=seg) * highpass(f_seg, LOW_CUT),
            n=seg,
        )
        # Augment teacher inputs with optional real noise + mic EQ / BT band-limit while keeping 48k studio target dry
        tnz = noise.sample(n_tp, seg, wrap=True)
        tsnr = torch.empty(n_tp, 1, device=dev).uniform_(2, 30)
        tx = tx + tnz / rms(tnz) * active_rms(tt) * 10 ** (-tsnr / 20) * (
            torch.rand(n_tp, 1, device=dev) < 0.60
        )
        t_eq = 10 ** (random_eq_db(n_tp, f_seg, knots=8, std_db=3.5) / 20)
        t_lp = torch.where(
            torch.rand(n_tp, 1, device=dev) < 0.40,
            torch.sigmoid(
                (torch.empty(n_tp, 1, device=dev).uniform_(6000.0, 12000.0) - f_seg)
                / 180.0
            ),
            1.0,
        )
        t_filt = torch.where(torch.rand(n_tp, 1, device=dev) < 0.50, t_eq, 1.0) * t_lp
        tx = torch.fft.irfft(torch.fft.rfft(tx, n=seg) * t_filt, n=seg)
        noisy[real_end : real_end + n_tp], target[real_end : real_end + n_tp] = tx, tt

    if clean_prob > 0:
        is_clean = coin(clean_prob)
        noisy = torch.where(is_clean, target, noisy)

    # Pure non-speech / distant-chatter zero-target rows (target == 0) so cafes/music/keyboards/wind stay silent
    if noise_only_prob > 0 and force_bandlim_hz is None:
        is_ns = coin(noise_only_prob)
        if is_ns.any():
            ns_bed = noise.sample(n, seg, wrap=True)
            ns_lvl = 10 ** (rand(-45, -18) / 20)
            ns_sig = ns_bed / rms(ns_bed) * ns_lvl + _wind_gusts(
                n, seg, ns_lvl, dev
            ) * coin(0.35)
            if events is not None:
                ns_ev = events.sample(n, seg)
                ns_sig = ns_sig + ns_ev / active_rms(ns_ev) * ns_lvl * 10 ** (
                    -rand(-3, 12) / 20
                ) * coin(0.50)
            # Distant reverberant babble with no primary talker (teaches model not to hallucinate words from cafes/TV)
            ns_sig = ns_sig + bab / rms(bab) * ns_lvl * 10 ** (
                -rand(0, 12) / 20
            ) * coin(0.35)
            ns_sig = torch.fft.irfft(
                (torch.fft.rfft(ns_sig, n=L) * mic + hiss) * lowpass, n=L
            )[:, :seg]
            noisy = torch.where(is_ns, ns_sig, noisy)
            target = torch.where(is_ns, torch.zeros_like(target), target)

    peak = torch.maximum(
        target.abs().amax(1, keepdim=True), torch.full((n, 1), 0.92, device=dev)
    )
    scale_safe = 0.92 / peak
    return noisy * scale_safe, target * scale_safe
