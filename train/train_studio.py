"""Train StudioNet: causal 48 kHz dual-path studio speech restoration model.

Two-stage distortion -> perception recipe (RE-USE 2026 + FINALLY + CleanMyMic real-call scenes):
  Stage 1 (regression + WavLM CNN content + high-band):
    python train/train_studio.py ~/data ~/experiments/studio-20261119/runs/stage1 \
      --steps 25000 --mic-irs ~/experiments/studio-20261119/data/surrey_mics_48k.npy \
      --real-pairs ~/experiments/realman-20261119/pairs/train
  Stage 2 (GAN studio polish with regression + WavLM fidelity anchor):
    python train/train_studio.py ~/data ~/experiments/studio-20261119/runs/stage2 \
      --steps 10000 --gan --init ~/experiments/studio-20261119/runs/stage1/best.pt \
      --mic-irs ~/experiments/studio-20261119/data/surrey_mics_48k.npy \
      --real-pairs ~/experiments/realman-20261119/pairs/train
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import nn

import dfn3_studio
from model import SR, load as load_ckpt
from studio_data import (
    EVENT_LEN,
    STUDIO_BEDS,
    STUDIO_CLEAN,
    STUDIO_CLEAN_POLISH,
    STUDIO_DEHISS,
    STUDIO_EVENTS,
    Pool,
    RealPairPool,
    TeacherPairPool,
    active_rms,
    align_rirs,
    by_distance,
    load_mic_irs,
    make_studio_batch,
)
from studio_net import STUDIO_PRESETS, StudioNet, StudioNetV2

SEG = 6 * SR
GAN_SEG = SR // 2


class STFTDisc(nn.Module):
    """EnCodec-style discriminator on one compressed complex STFT resolution."""

    def __init__(self, n_fft, ch=32):
        super().__init__()
        wn = nn.utils.parametrizations.weight_norm
        self.n_fft = n_fft
        self.convs = nn.ModuleList(
            [wn(nn.Conv2d(2, ch, (3, 9), padding=(1, 4)))]
            + [
                wn(
                    nn.Conv2d(
                        ch, ch, (3, 9), stride=(1, 2), dilation=(d, 1), padding=(d, 4)
                    )
                )
                for d in (1, 2, 4)
            ]
            + [wn(nn.Conv2d(ch, ch, (3, 3), padding=(1, 1)))]
        )
        self.post = wn(nn.Conv2d(ch, 1, (3, 3), padding=(1, 1)))
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, x):
        X = torch.stft(
            x,
            self.n_fft,
            self.n_fft // 4,
            window=self.window,
            normalized=True,
            return_complex=True,
        )
        Xc, _ = _safe_compress(X)
        h, feats = torch.stack([Xc.real, Xc.imag], 1).transpose(2, 3), []
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            for conv in self.convs:
                h = F.leaky_relu(conv(h), 0.2)
                feats.append(h.float())
            return self.post(h).float(), feats


def gan_losses(discs, y, t):
    """Generator side: hinge adversarial + relative feature matching, averaged over discriminators."""
    adv = fm = 0.0
    for d in discs:
        logit, feats = d(y)
        with torch.no_grad():
            _, real = d(t)
        adv = adv + F.relu(1.0 - logit).mean()
        fm = fm + sum(
            (a - b).abs().mean() / (b.abs().mean() + 1e-5) for a, b in zip(feats, real)
        ) / len(feats)
    return adv / len(discs), fm / len(discs)


class HighBandDisc(nn.Module):
    """2D log-spectrogram discriminator focused on 6-24 kHz (Xiph 2024/FINALLY finding: sharpens BWE air & sibilance)."""

    def __init__(self, n_fft=1024, ch=32):
        super().__init__()
        wn = nn.utils.parametrizations.weight_norm
        self.n_fft = n_fft
        self.lo_bin = int(
            round(6000.0 * n_fft / SR)
        )  # 6 kHz -> bin 128..513 (385 bins)
        self.register_buffer("window", torch.hann_window(n_fft))
        self.convs = nn.ModuleList(
            [
                wn(nn.Conv2d(1, ch, (3, 7), stride=(1, 2), padding=(1, 3))),
                wn(nn.Conv2d(ch, ch * 2, (3, 7), stride=(1, 2), padding=(1, 3))),
                wn(nn.Conv2d(ch * 2, ch * 2, (3, 5), stride=(1, 2), padding=(1, 2))),
                wn(nn.Conv2d(ch * 2, ch * 2, (3, 5), stride=(1, 2), padding=(1, 2))),
                wn(nn.Conv2d(ch * 2, ch * 2, (3, 3), padding=(1, 1))),
            ]
        )
        self.post = wn(nn.Conv2d(ch * 2, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        X = torch.stft(
            x,
            self.n_fft,
            self.n_fft // 4,
            window=self.window,
            normalized=True,
            return_complex=True,
        )
        hi = (_safe_abs(X[:, self.lo_bin :]) + 1e-5).log()  # [B, F_hi, T]
        h = hi.transpose(1, 2).unsqueeze(1)  # [B, 1, T, F_hi]
        feats = []
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            for conv in self.convs:
                h = F.leaky_relu(conv(h), 0.2)
                feats.append(h.float())
            return self.post(h).float(), feats


class PeriodDisc(nn.Module):
    """HiFi-GAN Multi-Period Waveform Discriminator: refines periodic vocal pitch harmonics for close-mic studio polish."""

    def __init__(self, period, ch=24):
        super().__init__()
        wn = nn.utils.parametrizations.weight_norm
        self.period = period
        self.convs = nn.ModuleList(
            [
                wn(nn.Conv2d(1, ch, (5, 1), (3, 1), padding=(2, 0))),
                wn(nn.Conv2d(ch, ch * 2, (5, 1), (3, 1), padding=(2, 0))),
                wn(nn.Conv2d(ch * 2, ch * 4, (5, 1), (3, 1), padding=(2, 0))),
                wn(nn.Conv2d(ch * 4, ch * 4, (5, 1), (3, 1), padding=(2, 0))),
                wn(nn.Conv2d(ch * 4, ch * 4, (3, 1), 1, padding=(1, 0))),
            ]
        )
        self.post = wn(nn.Conv2d(ch * 4, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        # x: [B, N] -> pad to multiple of period -> [B, 1, N//p, p]
        b, n = x.shape
        if n % self.period != 0:
            x = F.pad(x, (0, self.period - (n % self.period)), mode="reflect")
        h = x.view(b, 1, -1, self.period)
        feats = []
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            for conv in self.convs:
                h = F.leaky_relu(conv(h), 0.2)
                feats.append(h.float())
            return self.post(h).float(), feats


class WavLMConvLoss(nn.Module):
    """Frozen WavLM CNN feature-encoder + Layer-1 transformer perceptual loss (FINALLY + PASE 2025).

    Supervises 0-8 kHz phonetic content and speaker timbre at 16 kHz to prevent word deletions and voice drift.
    """

    def __init__(self, model_name="microsoft/wavlm-base-plus", device="cuda"):
        super().__init__()
        import torchaudio.functional as AF
        from transformers import WavLMModel

        self._resample = AF.resample
        full = WavLMModel.from_pretrained(model_name)
        self.fe = full.feature_extractor.to(device).eval()
        self.fp = full.feature_projection.to(device).eval()
        self.pos_conv = full.encoder.pos_conv_embed.to(device).eval()
        self.enc_ln = full.encoder.layer_norm.to(device).eval()
        self.layer0 = full.encoder.layers[0].to(device).eval()
        for mod in (self.fe, self.fp, self.pos_conv, self.enc_ln, self.layer0):
            for p in mod.parameters():
                p.requires_grad_(False)
        del full

    def _extract(self, wav_16k):
        x = wav_16k.unsqueeze(1)  # [B, 1, N]
        feats = []
        for layer in self.fe.conv_layers:
            x = layer(x)
            feats.append(x)
        # Transformer layer 1 (PASE: preserves fine speaker timbre + phonetic clarity)
        h = x.transpose(1, 2)
        h, _ = self.fp(h)
        h = h + self.pos_conv(h)
        h = self.enc_ln(h)
        h = self.layer0(h)[0]
        feats.append(h.transpose(1, 2))
        return feats

    def forward(self, y_norm, t_norm):
        y16 = self._resample(y_norm, SR, 16000)
        t16 = self._resample(t_norm, SR, 16000)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=y16.is_cuda):
            fy = self._extract(y16)
            with torch.no_grad():
                ft = self._extract(t16)
        loss = 0.0
        for a, b in zip(fy, ft):
            a_f, b_f = a.float(), b.float()
            loss = loss + (a_f - b_f).abs().mean() / (b_f.abs().mean().clamp(min=1e-4))
        return loss / len(fy)


def _safe_abs(Z):
    """Complex magnitude with smooth non-zero floor so backward never hits 0/0 NaN on exact silent bins."""
    return (Z.real.pow(2) + Z.imag.pow(2) + 1e-12).sqrt()


def _safe_compress(Z, c=0.3):
    mag = _safe_abs(Z)
    scale = mag.pow(c - 1.0)
    return torch.complex(Z.real * scale, Z.imag * scale), mag.pow(c)


class StudioLoss(nn.Module):
    """Multi-resolution complex+asymmetric STFT + 8-24 kHz high-band + third-octave tone + L1 + SNR + WavLM conv."""

    def __init__(
        self,
        asym=0.35,
        hb_weight=1.5,
        wav_weight=2.0,
        sdr_weight=0.15,
        tone_weight=0.5,
        ns_weight=2.0,
        wavlm_weight=0.0,
        wavlm=None,
        device="cuda",
    ):
        super().__init__()
        self.asym = asym
        self.hb_weight = hb_weight
        self.wav_weight = wav_weight
        self.sdr_weight = sdr_weight
        self.tone_weight = tone_weight
        self.ns_weight = ns_weight
        self.wavlm_weight = wavlm_weight
        self.wavlm = wavlm
        self.ffts = (256, 512, 1024, 2048, 4096)
        for n in self.ffts:
            self.register_buffer(f"w{n}", torch.hann_window(n, device=device))
        self.hb_lo = int(round(8000.0 * 1024 / SR))
        # Third-octave filterbank matrix [22 bands, 1025 bins] spanning 100 Hz -> 16 kHz for studio tone matching
        freqs = torch.fft.rfftfreq(2048, 1 / SR, device=device)
        edges = 100.0 * (2.0 ** (torch.arange(23, device=device) / 3.0))
        fb = torch.zeros(22, len(freqs), device=device)
        for i in range(22):
            m = (freqs >= edges[i]) & (freqs < edges[i + 1])
            if m.any():
                fb[i, m] = 1.0
        self.register_buffer("tone_fb", fb)

    def forward(self, y, t, x=None, use_wavlm=True):
        # Handle pure-noise zero-target rows safely: normalize by input level instead of target level
        has_speech = t.abs().amax(1) > 1e-4
        ref_rms = active_rms(t)
        if (~has_speech).any():
            in_rms = active_rms(x) if x is not None else torch.full_like(ref_rms, 0.05)
            ref_rms = torch.where(has_speech[:, None], ref_rms, in_rms)
        g = 0.05 / ref_rms
        yn, tn = y * g, t * g
        spec = 0.0
        hb = 0.0
        tone_l = 0.0
        for n in self.ffts:
            w = getattr(self, f"w{n}")
            Ys = torch.stft(
                yn, n, n // 4, window=w, normalized=True, return_complex=True
            )
            Ts = torch.stft(
                tn, n, n // 4, window=w, normalized=True, return_complex=True
            )
            Yc, Ym = _safe_compress(Ys)
            Tc, Tm = _safe_compress(Ts)
            d = Ym - Tm
            asym_w = torch.where(has_speech[:, None, None], self.asym, 0.0)
            mag_l = (d.pow(2) * (1.0 + asym_w * (d < 0))).mean()
            cplx_l = ((Yc.real - Tc.real).pow(2) + (Yc.imag - Tc.imag).pow(2)).mean()
            spec = spec + 400.0 * (mag_l + 0.5 * cplx_l)
            if n == 1024 and has_speech.any():
                y_hi = (_safe_abs(Ys[has_speech, self.hb_lo :]) + 1e-5).log()
                t_hi = (_safe_abs(Ts[has_speech, self.hb_lo :]) + 1e-5).log()
                dh = y_hi - t_hi
                hb = (dh.abs() * (1.0 + 0.5 * self.asym * (dh < 0))).mean()
            if n == 2048 and self.tone_weight > 0 and has_speech.any():
                Yp = Ys[has_speech].real.pow(2) + Ys[has_speech].imag.pow(2)
                Tp = Ts[has_speech].real.pow(2) + Ts[has_speech].imag.pow(2)
                te = Tp.sum(1, keepdim=True)
                act = (te >= te.amax(-1, keepdim=True) * 0.01).float()
                denom = act.sum(-1).clamp(min=1.0)
                yb = torch.einsum("kf,bft->bkt", self.tone_fb, Yp * act).sum(-1) / denom
                tb = torch.einsum("kf,bft->bkt", self.tone_fb, Tp * act).sum(-1) / denom
                td = 10.0 * ((yb + 1e-8).log10() - (tb + 1e-8).log10())
                tone_l = (td - td.mean(-1, keepdim=True)).abs().mean()
        spec = spec / len(self.ffts) * 4.0
        wav_l = (yn - tn).abs().mean()
        if has_speech.any():
            err_p = (yn[has_speech] - tn[has_speech]).pow(2).mean(1)
            ref_p = tn[has_speech].pow(2).mean(1).clamp(min=1e-6)
            sdr_l = 10.0 * ((err_p + 1e-6) / ref_p).log10().clamp(min=-25.0).mean()
        else:
            sdr_l = torch.tensor(-25.0, device=y.device)
        ns_l = 0.0
        if self.ns_weight > 0 and (~has_speech).any():
            # Push pure non-speech / distant-chatter residual below -55 dB relative to normalized input
            ns_pow = yn[~has_speech].pow(2).mean(1)
            ns_db = 10.0 * (ns_pow + 1e-12).log10()
            ns_l = (
                F.relu(ns_db + 81.0).mean() / 20.0
            )  # -26 dBFS input - 55 dB = -81 dBFS floor
        total = (
            spec
            + self.hb_weight * hb
            + self.wav_weight * wav_l
            + self.sdr_weight * (sdr_l + 25.0)
            + self.tone_weight * tone_l
            + self.ns_weight * ns_l
        )
        wlm = 0.0
        if (
            use_wavlm
            and self.wavlm is not None
            and self.wavlm_weight > 0
            and has_speech.any()
        ):
            wlm = self.wavlm(yn[has_speech], tn[has_speech])
            total = total + self.wavlm_weight * wlm
        return total, {
            "spec": float(spec.detach()),
            "hb": float(hb.detach()) if isinstance(hb, torch.Tensor) else 0.0,
            "tone": float(tone_l.detach()) if isinstance(tone_l, torch.Tensor) else 0.0,
            "wav": float(wav_l.detach()),
            "sdr": float(-sdr_l.detach()),
            "ns": float(ns_l.detach()) if isinstance(ns_l, torch.Tensor) else 0.0,
            "wlm": float(wlm.detach()) if isinstance(wlm, torch.Tensor) else 0.0,
        }


@torch.no_grad()
def spectral_db_metrics(y, t):
    """Full-band and 8-20 kHz mean absolute spectral error in dB."""
    w = torch.hann_window(1024, device=y.device)
    Ym = torch.stft(y, 1024, 256, window=w, return_complex=True).abs()
    Tm = torch.stft(t, 1024, 256, window=w, return_complex=True).abs()
    floor = Tm.amax((1, 2), keepdim=True) * 1e-3 + 1e-7
    diff_db = (20.0 * ((Ym + floor).log10() - (Tm + floor).log10())).abs()
    f = torch.fft.rfftfreq(1024, 1 / SR, device=y.device)
    hi = (f >= 8000) & (f <= 20000)
    hi_floor = Tm[:, hi].amax((1, 2), keepdim=True) * 1e-2 + 1e-7
    hi_db = (
        20.0 * ((Ym[:, hi] + hi_floor).log10() - (Tm[:, hi] + hi_floor).log10())
    ).abs()
    return float(diff_db.mean()), float(hi_db.mean()), float(y.abs().max())


def save_atomic(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.rename(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="root data dir containing packed5/ and packed6/")
    ap.add_argument("run", help="output run directory")
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument(
        "--seg-sec", type=int, default=4, help="training crop length in seconds"
    )
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument(
        "--init", help="checkpoint to initialize from (ignored when resuming)"
    )
    ap.add_argument(
        "--gan",
        action="store_true",
        help="Stage 2: add STFT + HighBand discriminators & polish clean pool",
    )
    ap.add_argument(
        "--gan-adv", type=float, default=1.0, help="adversarial loss weight"
    )
    ap.add_argument(
        "--gan-fm", type=float, default=3.0, help="feature-matching loss weight"
    )
    ap.add_argument(
        "--wavlm-weight",
        type=float,
        default=3.0,
        help="WavLM CNN encoder perceptual loss weight (0 disables)",
    )
    ap.add_argument("--asym", type=float, default=0.35)
    ap.add_argument(
        "--hold-every",
        type=int,
        default=8,
        help="run a 30s long-pause hold batch every K steps (0 disables)",
    )
    ap.add_argument("--mic-irs", help="path to packed Surrey mic IRs .npy")
    ap.add_argument("--real-pairs", help="path to RealMAN paired room train dir")
    ap.add_argument(
        "--teacher-pairs",
        help="path to verified offline teacher pairs dir (noisy/ + clean/, e.g. NemotronLabs RE-USE)",
    )
    ap.add_argument(
        "--teacher-ckpt",
        help="optional online teacher checkpoint (.pt) for student distillation (e.g. Pro -> Lite)",
    )
    ap.add_argument(
        "--distill-weight",
        type=float,
        default=0.5,
        help="distillation loss weight when --teacher-ckpt is provided",
    )
    ap.add_argument(
        "--extra-clean",
        help="optional extra packed studio speech pool prefix (e.g. hifitts_clean)",
    )
    ap.add_argument(
        "--extra-clean-weight",
        type=float,
        default=0.40,
        help="sampling share for --extra-clean",
    )
    ap.add_argument(
        "--noise-only-prob",
        type=float,
        default=0.10,
        help="share of pure non-speech / distant-chatter zero-target rows",
    )
    ap.add_argument(
        "--max-minutes",
        type=float,
        default=0.0,
        help="optional wall-clock budget in minutes",
    )
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--size", choices=("lite", "m", "l", "pro", "max"), default="l")
    ap.add_argument(
        "--arch",
        choices=("studio", "studio_v2", "dfn3_studio", "dfn3_studio_v2"),
        default="studio",
    )
    ap.add_argument(
        "--spk-dim",
        type=int,
        default=0,
        help="optional speaker embedding dim for FiLM conditioning (0 disables)",
    )
    ap.add_argument(
        "--no-freeze-stage1",
        action="store_true",
        help="for --arch studio_v2 --gan: do not freeze Stage-1 backbone",
    )
    ap.add_argument(
        "--base-ckpt",
        help="for --arch dfn3_studio: path to CleanMyMic Base (hold-10000.pt)",
    )
    ap.add_argument(
        "--base-lr-mult",
        type=float,
        default=0.25,
        help="LR multiplier on pretrained Base weights in dfn3_studio",
    )
    args = ap.parse_args()

    run = Path(args.run).expanduser()
    (run / "samples").mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(2)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    root = Path(args.data).expanduser()
    spec = lambda d: {root / p: w for p, w in d.items()}
    dehiss = {root / p for p in STUDIO_DEHISS}
    clean_spec = dict(STUDIO_CLEAN_POLISH if args.gan else STUDIO_CLEAN)
    clean_train_map = spec(clean_spec)
    clean_val_map = spec(STUDIO_CLEAN)
    if args.extra_clean and Path(f"{args.extra_clean}.npy").exists():
        ew = args.extra_clean_weight
        clean_train_map = {k: v * (1.0 - ew) for k, v in clean_train_map.items()} | {
            Path(args.extra_clean): ew
        }
        clean_val_map = {k: v * (1.0 - ew) for k, v in clean_val_map.items()} | {
            Path(args.extra_clean): ew
        }
        print(
            f"Added extra studio clean pool {args.extra_clean} (share={ew:.0%})",
            flush=True,
        )
    clean = Pool(clean_train_map, dev, dehissed=dehiss, memmap=True, split="train")
    clean_val = Pool(clean_val_map, dev, dehissed=dehiss, memmap=True, split="val")
    noise = Pool(spec(STUDIO_BEDS), dev, memmap=True, split="train")
    noise_val = Pool(spec(STUDIO_BEDS), dev, memmap=True, split="val")
    events = Pool(
        spec(STUDIO_EVENTS), dev, memmap=True, split="train", max_len=EVENT_LEN
    )
    events_val = Pool(
        spec(STUDIO_EVENTS), dev, memmap=True, split="val", max_len=EVENT_LEN
    )

    rirs = align_rirs(torch.from_numpy(np.load(root / "packed5" / "rirs.npy")).to(dev))
    rirs_train, rirs_val = rirs[:3800], rirs[3800:]
    rooms_train = by_distance(rirs_train)
    rooms_val = by_distance(rirs_val)
    mic_irs = load_mic_irs(args.mic_irs, device=dev) if args.mic_irs else None
    if mic_irs is not None:
        print(f"Loaded {len(mic_irs)} measured mic IRs from {args.mic_irs}", flush=True)
    real_pairs = RealPairPool(args.real_pairs, device=dev) if args.real_pairs else None
    if real_pairs is not None:
        print(
            f"Loaded {len(real_pairs)} real paired room clips from {args.real_pairs}",
            flush=True,
        )
    teacher_pairs = (
        TeacherPairPool(args.teacher_pairs, device=dev) if args.teacher_pairs else None
    )
    if teacher_pairs is not None:
        print(
            f"Loaded {len(teacher_pairs)} verified teacher pairs from {args.teacher_pairs}",
            flush=True,
        )
    teacher_model = None
    if args.teacher_ckpt:
        teacher_model, _ = load_ckpt(
            Path(args.teacher_ckpt).expanduser(), map_location=dev
        )
        teacher_model = teacher_model.to(dev).eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        print(
            f"Loaded frozen online teacher {args.teacher_ckpt} "
            f"({sum(p.numel() for p in teacher_model.parameters()) / 1e6:.2f}M, weight={args.distill_weight})",
            flush=True,
        )

    init_state = None
    if not (run / "last.pt").exists() and args.init:
        init_state = torch.load(
            Path(args.init).expanduser(), map_location=dev, weights_only=False
        )

    freeze_s1 = (
        args.arch in ("studio_v2", "dfn3_studio_v2")
        and args.gan
        and not args.no_freeze_stage1
    )
    if args.arch == "dfn3_studio":
        res_ch = (32, 64, 96) if args.size in ("l", "pro", "max") else (24, 48, 64)
        n_attn = 3 if args.size in ("l", "pro", "max") else 2
        if args.base_ckpt and not (run / "last.pt").exists() and not args.init:
            model = dfn3_studio.init_from_base(
                Path(args.base_ckpt).expanduser(), res_ch=res_ch, n_attn=n_attn
            ).to(dev)
            print(
                f"Initialized DFN3Studio from Base {args.base_ckpt} (spk={model.base.spk})",
                flush=True,
            )
        else:
            spk = 0
            if init_state is not None:
                spk = init_state.get("config", {}).get("spk", 0)
            elif (run / "last.pt").exists():
                spk = (
                    torch.load(run / "last.pt", map_location="cpu", weights_only=False)
                    .get("config", {})
                    .get("spk", 0)
                )
            model = dfn3_studio.DFN3Studio(res_ch=res_ch, n_attn=n_attn, spk=spk).to(
                dev
            )
        config = {
            "arch": "dfn3_studio",
            "res_ch": res_ch,
            "n_attn": n_attn,
            "spk": model.base.spk,
        }
    elif args.arch == "dfn3_studio_v2":
        s1_ch = (
            (32, 64, 96, 128) if args.size in ("pro", "max", "l") else (24, 48, 64, 96)
        )
        s1_blocks = 8 if args.size == "max" else (6 if args.size in ("pro", "l") else 4)
        ref_ch = (24, 48, 80, 96)
        ref_blocks = 3
        use_refiner = bool(args.gan)
        if args.base_ckpt and not (run / "last.pt").exists() and not args.init:
            model = dfn3_studio.init_v2_from_base(
                Path(args.base_ckpt).expanduser(),
                s1_ch=s1_ch,
                s1_blocks=s1_blocks,
                ref_ch=ref_ch,
                ref_blocks=ref_blocks,
                use_refiner=use_refiner,
            ).to(dev)
            print(
                f"Initialized DFN3StudioV2 from Base {args.base_ckpt} (spk={model.base.spk}, use_refiner={use_refiner})",
                flush=True,
            )
        else:
            icfg = {}
            if init_state is not None:
                icfg = init_state.get("config", {})
            elif (run / "last.pt").exists():
                icfg = torch.load(
                    run / "last.pt", map_location="cpu", weights_only=False
                ).get("config", {})
            s1_ch = tuple(icfg.get("s1_ch", s1_ch))
            s1_blocks = icfg.get("s1_blocks", s1_blocks)
            ref_ch = tuple(icfg.get("ref_ch", ref_ch))
            ref_blocks = icfg.get("ref_blocks", ref_blocks)
            spk = icfg.get("spk", 0)
            model = dfn3_studio.DFN3StudioV2(
                s1_ch=s1_ch,
                s1_blocks=s1_blocks,
                ref_ch=ref_ch,
                ref_blocks=ref_blocks,
                use_refiner=use_refiner or bool(icfg.get("use_refiner", False)),
                spk=spk,
            ).to(dev)
        config = {
            "arch": "dfn3_studio_v2",
            "s1_ch": s1_ch,
            "s1_blocks": s1_blocks,
            "ref_ch": ref_ch,
            "ref_blocks": ref_blocks,
            "use_refiner": model.use_refiner,
            "spk": model.base.spk,
        }
        if freeze_s1 and model.use_refiner:
            model.freeze_stage1()
    elif args.arch == "studio_v2":
        preset = STUDIO_PRESETS[args.size]
        s1_cfg = dict(preset["s1"]) | {"spk_dim": args.spk_dim}
        ref_cfg = dict(preset["ref"])
        if init_state is not None:
            icfg = dict(init_state.get("config", {}))
            if icfg.get("arch") == "studio":
                icfg.pop("arch", None)
                s1_cfg = icfg
            elif icfg.get("arch") == "studio_v2":
                s1_cfg = dict(icfg["s1_cfg"])
                ref_cfg = {
                    k: icfg[k]
                    for k in ("ref_ch", "ref_blocks", "ref_heads")
                    if k in icfg
                }
        model = StudioNetV2(s1_cfg=s1_cfg, **ref_cfg).to(dev)
        config = {"arch": "studio_v2", "s1_cfg": s1_cfg} | ref_cfg
        if freeze_s1:
            model.freeze_stage1()
    else:
        model_cfg = dict(STUDIO_PRESETS[args.size]["s1"]) | {"spk_dim": args.spk_dim}
        config = {"arch": "studio"} | model_cfg
        model = StudioNet(**model_cfg).to(dev)

    def set_train_mode():
        model.train()
        if freeze_s1 and isinstance(model, StudioNetV2):
            model.stage1.eval()
        elif (
            freeze_s1
            and isinstance(model, dfn3_studio.DFN3StudioV2)
            and model.use_refiner
        ):
            model.base.eval()
            model.restore.eval()
        # Keep pretrained DFN3 BatchNorm statistics frozen during fine-tuning (critical for Base stability!)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    wavlm = WavLMConvLoss(device=dev) if args.wavlm_weight > 0 else None
    crit = StudioLoss(
        asym=args.asym, wavlm_weight=args.wavlm_weight, wavlm=wavlm, device=dev
    ).to(dev)
    distill_crit = (
        StudioLoss(
            asym=0.2,
            hb_weight=1.0,
            wav_weight=2.0,
            sdr_weight=0.1,
            tone_weight=0.3,
            ns_weight=0.0,
            device=dev,
        ).to(dev)
        if teacher_model is not None
        else None
    )

    lr = args.lr or (1e-4 if args.gan else 3e-4)
    if args.arch == "dfn3_studio" or (
        args.arch == "dfn3_studio_v2" and not (freeze_s1 and model.use_refiner)
    ):
        param_groups = [
            {
                "params": [p for p in model.base.parameters() if p.requires_grad],
                "lr": lr * args.base_lr_mult,
                "lr_mult": args.base_lr_mult,
            },
            {
                "params": [
                    p
                    for p in list(model.restore.parameters())
                    + (
                        list(model.refiner.parameters())
                        if getattr(model, "refiner", None) is not None
                        else []
                    )
                    if p.requires_grad
                ],
                "lr": lr,
                "lr_mult": 1.0,
            },
        ]
    elif freeze_s1 and isinstance(model, (StudioNetV2, dfn3_studio.DFN3StudioV2)):
        param_groups = [
            {"params": list(model.refiner.parameters()), "lr": lr, "lr_mult": 1.0}
        ]
    else:
        param_groups = [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": lr,
                "lr_mult": 1.0,
            }
        ]
    opt = torch.optim.AdamW(param_groups, lr=lr, betas=(0.8, 0.99), weight_decay=1e-4)
    discs = nn.ModuleList(
        [STFTDisc(n) for n in (4096, 2048, 1024, 512, 256)]
        + [HighBandDisc(1024)]
        + [PeriodDisc(p) for p in (2, 3, 5, 7, 11)]
    ).to(dev)
    opt_d = torch.optim.AdamW(discs.parameters(), lr=1.5e-4, betas=(0.8, 0.99))

    step = 0
    best_val = float("inf")
    elapsed_prev = 0.0
    if (run / "last.pt").exists():
        st = torch.load(run / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        if "discs" in st:
            discs.load_state_dict(st["discs"])
            opt_d.load_state_dict(st["opt_d"])
        step = st["step"]
        best_val = st.get("best_val", float("inf"))
        elapsed_prev = st.get("elapsed_s", 0.0)
        print(f"Resumed {run} at step {step} (best_val={best_val:.4f})", flush=True)
    elif init_state is not None:
        if (
            isinstance(model, StudioNetV2)
            and init_state.get("config", {}).get("arch") == "studio"
        ):
            model.stage1.load_state_dict(init_state["model"])
            print(
                f"Loaded Stage-1 weights from {args.init} into StudioNetV2.stage1 "
                f"(step {init_state.get('step', '?')}, frozen={freeze_s1})",
                flush=True,
            )
        elif (
            isinstance(model, dfn3_studio.DFN3StudioV2)
            and model.use_refiner
            and not init_state.get("config", {}).get("use_refiner", False)
        ):
            missing, unexpected = model.load_state_dict(
                init_state["model"], strict=False
            )
            assert not unexpected and all(k.startswith("refiner.") for k in missing), (
                f"unexpected mismatch loading S1 into DFN3StudioV2: missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
            print(
                f"Loaded Stage-1 weights from {args.init} into DFN3StudioV2 (base+restore frozen={freeze_s1}, "
                f"step {init_state.get('step', '?')})",
                flush=True,
            )
        else:
            model.load_state_dict(init_state["model"])
            print(
                f"Initialized weights from {args.init} (step {init_state.get('step', '?')})",
                flush=True,
            )

    n_tot_p = sum(p.numel() for p in model.parameters()) / 1e6
    n_trn_p = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(
        f"{args.arch}-{args.size.upper()}: {n_tot_p:.2f}M params ({n_trn_p:.2f}M trainable)"
        + (
            f", discs {sum(p.numel() for p in discs.parameters()) / 1e6:.2f}M"
            if args.gan
            else ""
        ),
        flush=True,
    )

    # Fixed held-out validation views (never changed across steps)
    torch.manual_seed(0)
    val_views = {
        "mixed": make_studio_batch(
            clean_val,
            noise_val,
            rirs_val,
            16,
            SEG,
            events=events_val,
            rooms=rooms_val,
            mic_irs=mic_irs,
            hold=0.25,
            clean_prob=0.0,
            bandlim_prob=0.5,
        ),
        "bandlim_8k": make_studio_batch(
            clean_val,
            noise_val,
            rirs_val,
            16,
            SEG,
            events=events_val,
            rooms=rooms_val,
            mic_irs=mic_irs,
            hold=0.0,
            clean_prob=0.0,
            force_bandlim_hz=8000.0,
        ),
        "clean_id": make_studio_batch(
            clean_val,
            noise_val,
            rirs_val,
            16,
            SEG,
            events=events_val,
            rooms=rooms_val,
            clean_prob=1.0,
        ),
        "hold": make_studio_batch(
            clean_val,
            noise_val,
            rirs_val,
            3,
            30 * SR,
            events=events_val,
            rooms=rooms_val,
            mic_irs=mic_irs,
            hold=1.0,
            clean_prob=0.0,
            bandlim_prob=0.4,
        ),
        "non_speech": make_studio_batch(
            clean_val,
            noise_val,
            rirs_val,
            12,
            SEG,
            events=events_val,
            rooms=rooms_val,
            mic_irs=mic_irs,
            hold=0.0,
            clean_prob=0.0,
            noise_only_prob=1.0,
        ),
    }

    def run_val(cur_step):
        nonlocal best_val
        model.eval()
        metrics = {"step": cur_step}
        with torch.no_grad():
            for name, (vx, vt) in val_views.items():
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=dev == "cuda"
                ):
                    vy = model(vx).float()
                vloss, parts = crit(
                    vy, vt, x=vx, use_wavlm=(name not in ("hold", "non_speech"))
                )
                full_db, hi_db, peak = spectral_db_metrics(vy, vt)
                row = {
                    "loss": round(float(vloss), 4),
                    "spec": round(parts["spec"], 4),
                    "hb": round(parts["hb"], 4),
                    "wlm": round(parts["wlm"], 4),
                    "full_db": round(full_db, 3),
                    "hi8_20_db": round(hi_db, 3),
                    "peak": round(peak, 3),
                }
                if name == "non_speech":
                    in_p = vx.pow(2).mean(1).clamp(min=1e-12)
                    out_p = vy.pow(2).mean(1).clamp(min=1e-12)
                    row["atten_db"] = round(
                        float((10.0 * (in_p / out_p).log10()).mean()), 2
                    )
                metrics[name] = row
                if name in ("mixed", "bandlim_8k") and cur_step > 0:
                    for i in range(min(2, len(vx))):
                        for kind, arr in (("in", vx), ("out", vy), ("ref", vt)):
                            sf.write(
                                run
                                / "samples"
                                / f"{cur_step:06d}_{name}_{i}_{kind}.wav",
                                arr[i].cpu().numpy(),
                                SR,
                            )
        # Composite selection score (never selects on GAN loss): mixed + bandlim + clean identity + hold + non-speech
        sel = (
            metrics["mixed"]["loss"]
            + 0.7 * metrics["bandlim_8k"]["loss"]
            + 0.5 * metrics["clean_id"]["loss"]
            + 0.4 * metrics["hold"]["loss"]
            + 0.3 * metrics["non_speech"]["loss"]
        )
        metrics["sel"] = round(sel, 4)
        with open(run / "validation.jsonl", "a") as f:
            f.write(json.dumps(metrics) + "\n")
        print(
            f"VAL step={cur_step} sel={sel:.4f} mixed={metrics['mixed']['loss']:.3f} "
            f"bwe8k_hi={metrics['bandlim_8k']['hi8_20_db']:.2f}dB "
            f"clean={metrics['clean_id']['full_db']:.2f}dB hold={metrics['hold']['loss']:.3f} "
            f"ns_atten={metrics['non_speech']['atten_db']:.1f}dB",
            flush=True,
        )
        set_train_mode()
        return sel

    if step == 0:
        run_val(0)

    torch.manual_seed(args.seed + step)
    set_train_mode()
    wall_start = time.time()
    t0 = time.time()
    warmup = min(500, max(100, args.steps // 20))

    while step < args.steps:
        if (
            args.max_minutes > 0
            and (elapsed_prev + (time.time() - wall_start)) / 60.0 >= args.max_minutes
        ):
            print(
                f"Reached time budget {args.max_minutes} min at step {step}", flush=True
            )
            break

        # Cosine LR schedule with linear warmup (honors per-group lr_mult for pretrained Base vs RestoreHead)
        if step < warmup:
            cur_lr = lr * (step + 1) / warmup
        else:
            prog = (step - warmup) / max(1, args.steps - warmup)
            cur_lr = lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        for pg in opt.param_groups:
            pg["lr"] = cur_lr * pg.get("lr_mult", 1.0)

        is_hold = args.hold_every > 0 and (
            step % args.hold_every == args.hold_every - 1
        )
        if is_hold:
            x, t = make_studio_batch(
                clean,
                noise,
                rirs_train,
                3,
                30 * SR,
                events=events,
                rooms=rooms_train,
                mic_irs=mic_irs,
                hold=1.0,
                clean_prob=0.0,
                bandlim_prob=0.45,
                noise_only_prob=0.0,
            )
        else:
            x, t = make_studio_batch(
                clean,
                noise,
                rirs_train,
                args.batch,
                args.seg_sec * SR,
                events=events,
                rooms=rooms_train,
                mic_irs=mic_irs,
                real_pairs=real_pairs,
                teacher_pairs=teacher_pairs,
                hold=0.25,
                clean_prob=0.15 if args.gan else 0.12,
                bandlim_prob=0.50,
                noise_only_prob=args.noise_only_prob,
            )

        # Forward (BF16 in neural core, FP32 in STFT/loss)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            y = model(x).float()

        # On 30s hold batches skip WavLM to save VRAM/time
        rec, parts = crit(y, t, x=x, use_wavlm=(not is_hold))
        loss = rec
        extra = ""

        # Optional online student-teacher distillation (e.g. StudioNetV2-Pro -> StudioNetV2-Lite)
        if teacher_model is not None and args.distill_weight > 0 and not is_hold:
            with torch.no_grad():
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=dev == "cuda"
                ):
                    y_teach = teacher_model(x).float()
            d_l, _ = distill_crit(y, y_teach, x=x, use_wavlm=False)
            loss = loss + args.distill_weight * d_l
            extra += f" kd={d_l.item():.3f}"

        if args.gan and not is_hold:
            sp_idx = torch.nonzero(t.abs().amax(1) > 1e-4, as_tuple=False)[:, 0][:16]
            if len(sp_idx) >= 2:
                i = int(torch.randint(t.shape[1] - GAN_SEG, ()))
                yc, tc = y[sp_idx, i : i + GAN_SEG], t[sp_idx, i : i + GAN_SEG]
                sc = 0.1 / tc.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-4)
                yc, tc = yc * sc, tc * sc
                d_loss = sum(
                    F.relu(1.0 - d(tc)[0]).mean()
                    + F.relu(1.0 + d(yc.detach())[0]).mean()
                    for d in discs
                ) / len(discs)
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discs.parameters(), 2.0)
                opt_d.step()

                discs.requires_grad_(False)
                adv, fm = gan_losses(discs, yc, tc)
                discs.requires_grad_(True)
                loss = loss + args.gan_adv * adv + args.gan_fm * fm
                extra += (
                    f" adv={adv.item():.3f} fm={fm.item():.3f} d={d_loss.item():.3f}"
                )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        trainable = [p for p in model.parameters() if p.requires_grad]
        gn = torch.nn.utils.clip_grad_norm_(trainable, 1.5)
        if torch.isfinite(gn):
            opt.step()
        else:
            print(f"WARN step={step}: skipped non-finite grad norm", flush=True)
        step += 1

        if step % 100 == 0:
            sps = 100.0 / max(1e-3, time.time() - t0)
            mem = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0
            print(
                f"step={step}/{args.steps} lr={cur_lr:.2e} rec={rec.item():.3f} "
                f"(sp={parts['spec']:.2f} hb={parts['hb']:.2f} sdr={parts['sdr']:.1f}dB wlm={parts['wlm']:.2f}){extra} "
                f"{sps:.2f} it/s {mem:.1f}GB",
                flush=True,
            )
            t0 = time.time()

        if step % 500 == 0 or step == args.steps:
            elapsed = elapsed_prev + (time.time() - wall_start)
            ckpt = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "step": step,
                "best_val": best_val,
                "elapsed_s": elapsed,
                "config": config,
                "train_args": vars(args),
            }
            if args.gan:
                ckpt |= {"discs": discs.state_dict(), "opt_d": opt_d.state_dict()}
            save_atomic(ckpt, run / "last.pt")

        if step % args.val_every == 0 or step == args.steps:
            sel = run_val(step)
            if sel < best_val:
                best_val = sel
                ckpt = {
                    "model": model.state_dict(),
                    "step": step,
                    "best_val": best_val,
                    "config": config,
                    "train_args": vars(args),
                }
                save_atomic(ckpt, run / "best.pt")
                print(f"  -> new best.pt (sel={best_val:.4f})", flush=True)
            t0 = time.time()

    # Always save final checkpoint
    elapsed = elapsed_prev + (time.time() - wall_start)
    ckpt = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "step": step,
        "best_val": best_val,
        "elapsed_s": elapsed,
        "config": config,
        "train_args": vars(args),
    }
    if args.gan:
        ckpt |= {"discs": discs.state_dict(), "opt_d": opt_d.state_dict()}
    save_atomic(ckpt, run / "last.pt")
    print(f"Finished at step {step} in {elapsed / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
