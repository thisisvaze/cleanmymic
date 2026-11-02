"""The model: a causal per-frame network on the STFT.

v4: each frame goes through a conv encoder over frequency (481 -> 31 bands), a GRU across time in the middle, and a
mirrored decoder with skip connections (as v3). The head changed:
- every bin gets a bounded real gain (at most +2 dB), so it can't brighten hiss the way v3's unbounded mask did;
- the lowest DF_BINS bins (0-4.8 kHz) get a deep filter instead: DF_ORDER complex taps over the current and past
  frames of the linear spectrum, so it can follow the voice's harmonics under steady noise. v3 removed ~44 dB of noise
  in pauses but ~3.5 dB under speech; DeepFilterNet3, which has such a filter, beat even an oracle per-frame mask
  below 500 Hz on our desk noise;
- no generated residual (bandwidth extension comes back later as its own stage, only above a known cutoff).
The GRU also gets the lowest LOW bins' magnitudes at full resolution: the conv stack's bottleneck bands are ~800 Hz
wide, too coarse to tell fan rumble from a voice's fundamental. Convs only look along frequency, so streaming state
is the GRU's plus the last HIST input frames.
v5: optional lookahead. With look > 0 the output is frame t - look: the encoder sees that frame and the look frames
after it (stacked as channels), and the deep filter's taps span both sides of it, as in DeepFilterNet3 (look 2).
Each frame of lookahead adds 10 ms of latency. v4 kept noise in soft syllables and the gaps between words, where a
causal model can't yet tell a word's start or end from noise.
Also optional (v6 A/B): past frames stacked into the encoder input as well (convs otherwise see one frame), and a
per-bin noise-floor feature: how far each bin sits above its lowest 50 ms level of the last 1.5 s (minimum
statistics). The encoder works one frame at a time and the GRU sees only 31 bands, so nothing tracked the floor of
narrow steady noise (the MacBook's fan tones at 1-2 kHz), which DeepFilterNet3 gets from per-bin running norms.
And a per-band memory: a small GRU over time, shared by the 121 bands (~200 Hz each) of encoder level 2, whose
output is added to that level's features (as in DPCRN / GTCRN's per-band temporal paths).

Run this file to check that streaming (what the Mac app does) matches offline (what training does).
"""
import torch
import torch.nn.functional as F
from torch import nn

SR, WIN, HOP = 48000, 960, 480  # 20 ms window, 10 ms hop -> 20 ms algorithmic latency, + 10 ms per lookahead frame
BINS = WIN // 2 + 1
C = 0.3  # magnitude compression exponent
DF_BINS, DF_ORDER = 96, 5  # deep filter on 0-4.8 kHz: 5 taps over the newest input frame and the 4 before it
HIST = DF_ORDER - 1  # input frames carried between steps (also caps the lookahead)
LOW = 32  # bins (0-1.6 kHz) whose magnitudes reach the GRU at full resolution
MAX_GAIN = 1.25  # +2 dB
CH = (16, 24, 32, 48, 64)  # encoder channels per level (train.py --size multiplies them and the GRU)
FLOOR = 150  # frames (1.5 s) the noise floor is the minimum over: longer than most runs of speech without a pause
BIG = 100.0  # floor history is stored minus this, so the all-zero state the Mac app starts with means "no level yet"
BAND_LEVEL = 2  # encoder level of the per-band GRU: 121 bands of ~200 Hz


def compress(X):
    return X * (X.abs() + 1e-8) ** (C - 1)


def decompress(Y):
    return Y * (Y.abs() + 1e-8) ** (1 / C - 1)


def _power(re, im, p):  # (re, im) * |.|^(p - 1) on real tensors: Core ML has no complex dtype
    s = (re * re + im * im + 1e-16) ** ((p - 1) / 2)
    return re * s, im * s


class Model(nn.Module):
    def __init__(self, ch=CH, hidden=256, layers=2, df=True, look=0, past=0, floor=False, band=False):
        super().__init__()
        assert 0 <= look and 0 <= past and look + past <= HIST
        self.df, self.look, self.past, self.floor, self.band = df, look, past, floor, band
        # 2D convs with (1, 5) kernels on [B, C, T, F]: frequency only, and no reshapes that Core ML can't trace
        self.enc = nn.ModuleList(nn.Conv2d(a, b, (1, 5), stride=(1, 1 if i == 0 else 2), padding=(0, 2))
                                 for i, (a, b) in enumerate(zip((2 * (look + 1 + past) + floor,) + ch, ch)))
        self.dec = nn.ModuleList(nn.ConvTranspose2d(a, b, (1, 5), stride=(1, 2), padding=(0, 2))
                                 for a, b in zip(ch[:0:-1], ch[-2::-1]))
        self.bands = ch[-1], (BINS - 1) // 2 ** (len(ch) - 1) + 1  # 64 channels x 31 bands
        self.inp = nn.Linear(self.bands[0] * self.bands[1] + LOW, hidden)
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True)
        self.mid = nn.Linear(hidden, self.bands[0] * self.bands[1])
        self.out = nn.Conv2d(ch[0], 1 + 2 * DF_ORDER * df, (1, 5), padding=(0, 2))  # gain [, taps re, taps im]
        nn.init.zeros_(self.out.weight)  # start as passthrough: gain 1, the output frame's tap 1, other taps 0
        with torch.no_grad():
            self.out.bias.zero_()
            self.out.bias[0] = torch.logit(torch.tensor(1 / MAX_GAIN))
            if df:
                self.out.bias[1 + look] = torch.atanh(torch.tensor(1 / MAX_GAIN))
        if band:
            self.band_gru = nn.GRU(ch[BAND_LEVEL], ch[BAND_LEVEL], batch_first=True)
            self.band_out = nn.Linear(ch[BAND_LEVEL], ch[BAND_LEVEL])
            nn.init.zeros_(self.band_out.weight)  # what it adds starts at 0: the model starts unchanged
            nn.init.zeros_(self.band_out.bias)
        self.register_buffer("window", torch.hann_window(WIN).sqrt())

    @property
    def state_names(self):
        """step()'s state arguments, in the order it returns them (export.py names Core ML's inputs after them)."""
        return ["h", "buf"] + ["fl"] * self.floor + ["hb"] * self.band

    def step(self, x, h=None, buf=None, fl=None, hb=None):
        """x: compressed noisy STFT frames [B, T, 2*BINS] as (re, im); buf: the HIST input frames before them, the
        same way (zeros at the start); fl, hb: floor history and per-band GRU state (if used). Output frame t is the
        clean estimate of input frame t - look. Returns compressed clean frames, then the new state (state_names)."""
        if buf is None:
            buf = x.new_zeros(x.shape[0], HIST, x.shape[2])
        hist = torch.cat([buf, x], 1)
        back = lambda j: hist[:, HIST - j:-j] if j else hist[:, HIST:]  # frames t - j (negative ends: no traced lengths)
        frames = list(range(self.look, -1, -1)) + list(range(self.look + 1, self.look + 1 + self.past))
        a = [p for j in frames for p in back(j).chunk(2, -1)]
        if self.floor:
            level, fl = self._above_floor(hist, fl)
            a.append(level)
        a, skips = torch.stack(a, 1), []
        for i, conv in enumerate(self.enc):
            a = F.elu(conv(a))
            if self.band and i == BAND_LEVEL:
                a, hb = self._band(a, hb)
            skips.append(a)
        x_re, x_im = back(self.look).chunk(2, -1)  # the output frame
        low = (x_re[..., :LOW] ** 2 + x_im[..., :LOW] ** 2 + 1e-12).sqrt()
        z, h = self.gru(F.elu(self.inp(torch.cat([a.transpose(1, 2).flatten(2), low], -1))), h)
        a = F.elu(self.mid(z)).unflatten(-1, self.bands).transpose(1, 2)
        for deconv, skip in zip(self.dec, skips[:0:-1]):
            a = F.elu(deconv(a + skip))
        o = self.out(a + skips[0])  # [B, channels, T, BINS]
        g = MAX_GAIN * torch.sigmoid(o[:, 0])
        X_re, X_im = _power(x_re, x_im, 1 / C)  # the linear spectrum
        Y_re, Y_im = g * X_re, g * X_im
        if self.df:
            h_re, h_im = _power(hist[..., :DF_BINS], hist[..., BINS:BINS + DF_BINS], 1 / C)
            c = MAX_GAIN * torch.tanh(o[:, 1:, :, :DF_BINS])
            d_re = d_im = 0
            for i in range(DF_ORDER):  # tap i multiplies the input frame i steps back from the newest
                p_re, p_im = (h_re[:, HIST:], h_im[:, HIST:]) if i == 0 else (h_re[:, HIST - i:-i], h_im[:, HIST - i:-i])
                c_re, c_im = c[:, i], c[:, DF_ORDER + i]
                d_re = d_re + c_re * p_re - c_im * p_im
                d_im = d_im + c_re * p_im + c_im * p_re
            Y_re = torch.cat([d_re, Y_re[..., DF_BINS:]], -1)
            Y_im = torch.cat([d_im, Y_im[..., DF_BINS:]], -1)
        out = torch.cat(_power(Y_re, Y_im, C), -1), h, hist[:, -HIST:]
        return out + ((fl,) if self.floor else ()) + ((hb,) if self.band else ())

    def _band(self, a, hb):
        """Features [B, C, T, F] plus what the per-band GRU has heard in each band over time. The band count is a
        constant: sizes read off traced tensors break the Core ML conversion."""
        z, hb = self.band_gru(a.permute(0, 3, 2, 1).flatten(0, 1), hb)  # [B * F, T, C]: one sequence per band
        return a + self.band_out(z).unflatten(0, (-1, (BINS - 1) // 2 ** BAND_LEVEL + 1)).permute(0, 3, 2, 1), hb

    @staticmethod
    def _above_floor(hist, fl):
        """Each new frame's per-bin level above the noise floor (units of 20 dB, clamped to -20..+60 dB). The floor
        is the lowest 50 ms mean power of the last FLOOR frames; frames that are all zero (the empty buf at the start)
        don't count. Returns the feature [B, T, BINS] and the new floor history [B, FLOOR - 1, BINS]."""
        if fl is None:
            fl = hist.new_zeros(hist.shape[0], FLOOR - 1, BINS)
        re, im = hist.chunk(2, -1)
        p = (re * re + im * im) ** (1 / C)  # |X|^2 from the compressed spectrum (|X|^C)
        seen = (p.sum(-1, keepdim=True) > 0).float()
        num = F.avg_pool1d((p * seen).transpose(1, 2), DF_ORDER, 1)  # 50 ms means ending at each new frame
        den = F.avg_pool1d(seen.transpose(1, 2), DF_ORDER, 1)
        mean = torch.where(den > 0, (num / den.clamp(min=0.1) + 1e-10).log(), BIG).transpose(1, 2)
        seq = torch.cat([fl + BIG, mean], 1)
        floor = -F.max_pool1d(-seq.transpose(1, 2), FLOOR, 1).transpose(1, 2)
        level = ((p[:, HIST:] + 1e-10).log() - floor) * (10 / 2.302585 / 20)
        return level.clamp(-1, 3), seq[:, -(FLOOR - 1):] - BIG

    def forward(self, wav):
        X = compress(torch.stft(wav, WIN, HOP, window=self.window, pad_mode="constant", return_complex=True))
        x = F.pad(torch.cat([X.real, X.imag], 1).transpose(1, 2), (0, 0, 0, self.look))  # silence after the end
        y = self.step(x)[0][:, self.look:]  # aligned with the input again
        Y = decompress(torch.complex(*y.transpose(1, 2).chunk(2, 1)))
        return torch.istft(Y, WIN, HOP, window=self.window, length=wav.shape[-1])


def stream(model, wav):
    """Reference real-time loop: 480 samples in, 480 out. Output lags input by HOP * (1 + look) samples in this
    array (20 ms end to end at look 0, counting the frame being collected). The Mac app mirrors this."""
    buf, tail, state, out = torch.zeros(WIN), torch.zeros(HOP), {}, []
    for i in range(0, len(wav) - HOP + 1, HOP):
        buf = torch.cat([buf[HOP:], wav[i:i + HOP]])
        X = compress(torch.fft.rfft(buf * model.window))
        y, *state = model.step(torch.cat([X.real, X.imag])[None, None], **state)
        state = dict(zip(model.state_names, state))
        frame = torch.fft.irfft(decompress(torch.complex(*y[0, 0].chunk(2))), n=WIN) * model.window
        out.append(tail + frame[:HOP])
        tail = frame[HOP:]
    return torch.cat(out)


def load(path, map_location="cpu"):
    """Model from a checkpoint; v4+ checkpoints carry their constructor args. Studio-stage ones (train.py --arch
    dfn3) are DeepFilterNet3's network: dfn3.py."""
    state = torch.load(path, map_location=map_location)
    if state.get("config", {}).get("arch") == "dfn3":
        import dfn3
        return dfn3.load(path), state
    model = Model(**state.get("config", {}))
    model.load_state_dict(state["model"])
    return model, state


def widened(state, model):
    """Weights of a model for one with more inputs or modules (past frames, floor feature, per-band GRU): new input
    channels come after the old ones and start at 0, so it starts as the same model."""
    state = dict(state)
    w, n = state["enc.0.weight"], model.enc[0].in_channels
    state["enc.0.weight"] = torch.cat([w, w.new_zeros(w.shape[0], n - w.shape[1], *w.shape[2:])], 1)
    for k, v in model.state_dict().items():  # new modules (the per-band GRU) keep their init: they add 0 at first
        state.setdefault(k, v)
    return state


def delayed(state, look):
    """Weights of a look-0 model for one with lookahead: it starts as the same model run look frames later. The
    future input channels and taps start at 0; old tap i (look + i in the new window) is lost past the window's end."""
    state = dict(state)
    w = state["enc.0.weight"]
    state["enc.0.weight"] = torch.cat([w, w.new_zeros(w.shape[0], 2 * look, *w.shape[2:])], 1)
    for k in ("out.weight", "out.bias"):
        old = state[k]
        if len(old) > 1:  # gain, taps re, taps im
            state[k] = torch.zeros_like(old)
            state[k][0] = old[0]
            for part in (1, 1 + DF_ORDER):
                state[k][part + look:part + DF_ORDER] = old[part:part + DF_ORDER - look]
    return state


if __name__ == "__main__":
    torch.manual_seed(0)
    t = torch.arange(3 * SR) / SR
    wav = torch.randn(3 * SR) * 0.1 * (1.1 + torch.sin(2 * torch.pi * 0.7 * t))  # level swings: the floor moves
    for cfg in ({}, {"df": False}, {"look": 2}, {"df": False, "look": 2}, {"past": 2, "floor": True}, {"band": True},
                {"look": 2, "past": 2, "floor": True, "band": True}):
        df, look = cfg.get("df", True), cfg.get("look", 0)
        model = Model(**cfg).eval()
        with torch.no_grad():
            model.out.weight.normal_(0, 0.01)  # not passthrough, so every path is exercised
            model.enc[0].weight.normal_(0, 0.1)  # and the encoder reads every input channel
            if model.band:
                model.band_out.weight.normal_(0, 0.1)
            offline, streamed = model(wav[None])[0], stream(model, wav)
            passthrough = Model(**cfg).eval()(wav[None])[0]
        lag = HOP * (1 + look)
        err = (streamed[lag:] - offline[:len(streamed) - lag]).abs().max().item()
        assert err < 1e-4, f"streaming != offline (max diff {err})"
        p_err = (passthrough - wav)[WIN:-WIN].abs().max().item()
        assert p_err < 1e-3, f"untrained model is not passthrough (max diff {p_err})"
        if look:  # a look-0 model whose dropped taps are 0, moved to this lookahead, gives the same output
            with torch.no_grad():
                m0 = Model(df=df).eval()
                m0.out.weight.normal_(0, 0.01)
                if df:
                    for part in (1, 1 + DF_ORDER):
                        m0.out.weight[part + DF_ORDER - look:part + DF_ORDER] = 0
                model = Model(**cfg).eval()
                model.load_state_dict(widened(delayed(m0.state_dict(), look), model))
                d_err = (model(wav[None])[0] - m0(wav[None])[0])[SR // 2:].abs().max().item()
            assert d_err < 1e-4, f"delayed()/widened() changed the model (max diff {d_err})"
        elif cfg.get("past") or cfg.get("floor") or cfg.get("band"):  # new parts start switched off: same output
            with torch.no_grad():
                m0 = Model(df=df).eval()
                m0.out.weight.normal_(0, 0.01)
                model = Model(**cfg).eval()  # fresh: new modules at their init, as in training
                model.load_state_dict(widened(m0.state_dict(), model))
                w_err = (model(wav[None])[0] - m0(wav[None])[0]).abs().max().item()
            assert w_err < 1e-5, f"widened() changed the model (max diff {w_err})"
        macs = sum(m.weight.numel() for n, m in model.named_modules() if isinstance(m, nn.Linear) and n != "band_out")
        macs += sum(p.numel() for n, p in model.gru.named_parameters() if "weight" in n)
        if model.band:  # once per band at its level
            macs += (sum(p.numel() for n, p in model.band_gru.named_parameters() if "weight" in n)
                     + model.band_out.weight.numel()) * ((BINS - 1) // 2 ** BAND_LEVEL + 1)
        a = torch.zeros(1, model.enc[0].in_channels, 1, BINS)
        for conv in model.enc:  # conv MACs = weights x output length
            a = conv(a)
            macs += conv.weight.numel() * a.shape[-1]
        for deconv in model.dec:  # transposed conv MACs = weights x input length
            macs += deconv.weight.numel() * a.shape[-1]
            a = deconv(a)
        macs += model.out.weight.numel() * BINS
        print(f"ok ({cfg or 'v4'}): streaming matches offline (max diff {err:.1e}), passthrough at init, "
              f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, {macs * SR / HOP / 1e9:.2f} GMAC/s")
