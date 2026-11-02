"""DeepFilterNet3 as a streaming model for the app's second engine: one 10 ms frame in, one out, with explicit state,
like model.py, so export.py turns it into Core ML and the Mac app runs it the same way as ours.

The network is DeepFilterNet3's own (Schröter et al., github.com/Rikorose/DeepFilterNet, MIT OR Apache-2.0,
Copyright (c) 2021 Hendrik Schröter): its layers are copied below with their parameter names, so its released
checkpoint (DeepFilterNet3.zip: checkpoints/model_120.ckpt.best) loads as is, and the values of its config.ini are
fixed here. Its features (ERB levels and unit-normed spectrum with running means) and its STFT scaling are ported
from libDF. What is new is only the frame-by-frame driver.

Frames in and out are numpy-scaled rfft frames of the WINDOW-ed signal, as (re, im), not compressed. DFN3 looks two
frames ahead (conv and deep-filter lookahead), so output frame t comes out with input frame t + 2: 40 ms in all.

Ours on top, for the studio stage (train.py --arch dfn3 --studio): a boost head. DFN3's band gains stop at 1 (it
only removes), so it can't undo a dull or thin mic; conv0_gain scales them by up to BOOST either way. It starts at 0
(a factor of 1), so a DFN checkpoint without it runs exactly as DFN3. forward() is the training path: a whole clip at
once, computing what step() does frame by frame.

    uv run train/dfn3.py check <model_120.ckpt.best> in.wav dfn_out.wav   # vs DFN's enhance() of in.wav
    uv run train/dfn3.py selftest <model_120.ckpt.best>                    # forward() vs the streaming path
"""
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

SR, FFT, HOP = 48_000, 960, 480
BINS = FFT // 2 + 1
NB_ERB, NB_DF, DF_ORDER, LOOK = 32, 96, 5, 2
CH, EMB, HIDDEN = 64, 64 * 32 // 4, 256  # conv_ch, encoder embedding (conv_ch * nb_erb / 4), GRU size
ALPHA = 0.99  # libDF's norm alpha: exp(-hop / sr / norm_tau) rounded as DFN does
ERB_INIT = torch.linspace(-60, -90, NB_ERB)  # libDF's starting running means; the state holds the offset from these,
SPEC_INIT = torch.linspace(0.001, 0.0001, NB_DF)  # so the app's all-zero start state is DFN's start
WNORM = 1 / (FFT ** 2 / (2 * HOP))  # libDF scales its rfft by this; its inverse rfft is unnormalized
BOOST = 4.0  # the boost head scales band gains within [1/BOOST, BOOST]: up to +12 dB, for bands a mic lost
WINDOW = torch.sin(0.5 * math.pi * torch.sin(0.5 * math.pi * (torch.arange(FFT, dtype=torch.float64) + 0.5) / (FFT // 2)) ** 2).float()


def erb_widths(sr=SR, fft=FFT, nb=NB_ERB, min_nb=2):
    """libDF's erb_fb(): bins per ERB band."""
    freq2erb = lambda f: 9.265 * math.log1p(f / (24.7 * 9.265))
    erb2freq = lambda e: 24.7 * 9.265 * (math.exp(e / 9.265) - 1)
    step, widths, prev, over = freq2erb(sr // 2) / nb, [], 0, 0
    for i in range(1, nb + 1):
        fb = math.floor(erb2freq(i * step) / (sr / fft) + 0.5)  # Rust's round(): halves away from zero
        n = fb - prev - over
        over, n = (min_nb - n, min_nb) if n < min_nb else (0, n)
        widths.append(n)
        prev = fb
    widths[-1] += 1  # fft/2 + 1 bins
    widths[-1] -= max(0, sum(widths) - (fft // 2 + 1))
    return widths


def erb_fb(widths, inverse=False):
    fb = torch.zeros(sum(widths), len(widths))
    for i, (b, w) in enumerate(zip(np.cumsum([0] + widths[:-1]).tolist(), widths)):
        fb[b:b + w, i] = 1
    return fb.t() if inverse else fb / fb.sum(0)


# ---- DeepFilterNet3's layers (df/modules.py, df/deepfilternet3.py), with DFN3's config ----

class Conv2dNormAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, fstride=1, separable=True, activation_layer=nn.ReLU):
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        layers = [nn.ConstantPad2d((0, 0, kernel_size[0] - 1, 0), 0.0)] if kernel_size[0] > 1 else []
        groups = math.gcd(in_ch, out_ch) if separable else 1  # 1x1 convs stay grouped (per channel), as in DFN
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, padding=(0, kernel_size[1] // 2), stride=(1, fstride),
                                groups=groups, bias=False))
        if groups > 1 and max(kernel_size) > 1:
            layers.append(nn.Conv2d(out_ch, out_ch, 1, bias=False))
        super().__init__(*layers, nn.BatchNorm2d(out_ch), activation_layer())

    def frame(self, x):
        """x: the kernel's time window of inputs [B, C, kt, F] -> [B, C', 1, F] (the padding is the caller's state)."""
        for m in self:
            if not isinstance(m, nn.ConstantPad2d):
                x = m(x)
        return x


class ConvTranspose2dNormAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, fstride=1):
        groups = math.gcd(in_ch, out_ch)
        super().__init__(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, padding=(0, kernel_size[1] // 2),
                               output_padding=(0, kernel_size[1] // 2), stride=(1, fstride), groups=groups, bias=False),
            nn.Conv2d(out_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU())


class GroupedLinearEinsum(nn.Module):
    def __init__(self, input_size, hidden_size, groups=1):
        super().__init__()
        self.groups, self.ws = groups, input_size // groups
        self.weight = nn.Parameter(torch.zeros(groups, input_size // groups, hidden_size // groups))

    def forward(self, x):  # [B, T, I] -> [B, T, H]; a broadcast matmul instead of DFN's einsum (same math), and
        return (x.unflatten(-1, (self.groups, 1, self.ws)) @ self.weight).flatten(2)  # no traced shapes for Core ML


class SqueezedGRU_S(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None, num_layers=1, linear_groups=16):
        super().__init__()
        self.linear_in = nn.Sequential(GroupedLinearEinsum(input_size, hidden_size, linear_groups), nn.ReLU())
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.linear_out = (nn.Sequential(GroupedLinearEinsum(hidden_size, output_size, linear_groups), nn.ReLU())
                           if output_size else nn.Identity())

    def forward(self, x, h=None):
        y, h = self.gru(self.linear_in(x), h)
        return self.linear_out(y), h


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.erb_conv0 = Conv2dNormAct(1, CH, (3, 3))
        self.erb_conv1 = Conv2dNormAct(CH, CH, (1, 3), fstride=2)
        self.erb_conv2 = Conv2dNormAct(CH, CH, (1, 3), fstride=2)
        self.erb_conv3 = Conv2dNormAct(CH, CH, (1, 3), fstride=1)
        self.df_conv0 = Conv2dNormAct(2, CH, (3, 3))
        self.df_conv1 = Conv2dNormAct(CH, CH, (1, 3), fstride=2)
        self.df_fc_emb = nn.Sequential(GroupedLinearEinsum(CH * NB_DF // 2, EMB, groups=32), nn.ReLU())
        self.emb_gru = SqueezedGRU_S(EMB, HIDDEN, output_size=EMB, num_layers=1)
        self.lsnr_fc = nn.Sequential(nn.Linear(EMB, 1), nn.Sigmoid())  # loaded, unused (no lsnr gating, as enhance())


class ErbDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_gru = SqueezedGRU_S(EMB, HIDDEN, output_size=EMB, num_layers=2)
        self.conv3p = Conv2dNormAct(CH, CH, 1)
        self.convt3 = Conv2dNormAct(CH, CH, (1, 3))
        self.conv2p = Conv2dNormAct(CH, CH, 1)
        self.convt2 = ConvTranspose2dNormAct(CH, CH, (1, 3), fstride=2)
        self.conv1p = Conv2dNormAct(CH, CH, 1)
        self.convt1 = ConvTranspose2dNormAct(CH, CH, (1, 3), fstride=2)
        self.conv0p = Conv2dNormAct(CH, CH, 1)
        self.conv0_out = Conv2dNormAct(CH, 1, (1, 3), activation_layer=nn.Sigmoid)
        self.conv0_gain = nn.Conv2d(CH, 1, (1, 3), padding=(0, 1))  # ours: the boost head, 0 at the start (DFN3)
        nn.init.zeros_(self.conv0_gain.weight)
        nn.init.zeros_(self.conv0_gain.bias)

    def gains(self, h):
        """Band gains from the last level's features [B, CH, T, NB_ERB]: DFN3's sigmoid times our boost."""
        return (self.conv0_out(h) * torch.exp(math.log(BOOST) * torch.tanh(self.conv0_gain(h))))[:, 0]


class DfDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.df_convp = Conv2dNormAct(CH, DF_ORDER * 2, (5, 1))
        self.df_gru = SqueezedGRU_S(EMB, HIDDEN, num_layers=2, linear_groups=8)  # DFN leaves SqueezedGRU_S's default
        self.df_skip = GroupedLinearEinsum(EMB, HIDDEN, groups=16)
        self.df_out = nn.Sequential(GroupedLinearEinsum(HIDDEN, NB_DF * DF_ORDER * 2, groups=16), nn.Tanh())
        self.df_fc_a = nn.Sequential(nn.Linear(HIDDEN, 1), nn.Sigmoid())  # loaded, unused by DFN3's forward


# ---- the streaming model ----

class DFN3(nn.Module):
    """DFN3's DfNet (deepfilternet3.py), run one frame at a time. State: the feature normalizers' running means, the
    last 2 input feature frames (the 3-frame input convs), the last 4 c0 frames (the 5-frame DF pathway conv), the
    last 4 spectra (the output frame is 2 back; deep filtering spans it +-2), and the three GRUs."""

    state_names = ["s_erb", "s_spec", "f_erb", "f_spec", "c0", "spec", "h_enc", "h_erb", "h_df"]
    window = WINDOW

    def __init__(self, spk=0):
        super().__init__()
        widths = erb_widths()
        self.register_buffer("erb_fb", erb_fb(widths))
        self.enc, self.erb_dec, self.df_dec = Encoder(), ErbDecoder(), DfDecoder()
        self.mask = nn.Module()
        self.mask.register_buffer("erb_inv_fb", erb_fb(widths, inverse=True))
        self.spk = spk
        if spk:  # ours, the enrolled voice: a speaker embedding [B, spk] scales and shifts the encoder's embedding before
            # and after its GRU. No biases and the last layers at 0: it starts as the plain model, and a zero
            # embedding (nobody enrolled) always runs it as the plain model.
            film = lambda: nn.Sequential(nn.Linear(spk, 256, bias=False), nn.ReLU(), nn.Linear(256, 2 * EMB, bias=False))
            self.film_in, self.film_out = film(), film()
            for f in (self.film_in, self.film_out):
                nn.init.zeros_(f[2].weight)

    @staticmethod
    def film(layer, emb, spk):
        """emb [B, T, EMB] scaled and shifted by what layer makes of spk [B, S]."""
        g, b = layer(spk)[:, None].chunk(2, -1)
        return emb * (1 + g) + b

    def initial_state(self, b=1):
        z = lambda *s: torch.zeros(b, *s)
        return {"s_erb": z(NB_ERB), "s_spec": z(NB_DF), "f_erb": z(1, 2, NB_ERB), "f_spec": z(2, 2, NB_DF),
                "c0": z(CH, 4, NB_DF), "spec": z(4, 2 * BINS),
                "h_enc": torch.zeros(1, b, HIDDEN), "h_erb": torch.zeros(2, b, HIDDEN), "h_df": torch.zeros(2, b, HIDDEN)}

    def step(self, x, spk=None, **state):
        """x: one noisy frame [B, 1, 2*BINS] (re, im); spk: the enrolled voice [B, self.spk], if any. Returns the clean
        frame for the input 2 frames back, then the new state (state_names); missing state starts as at the beginning
        of a stream."""
        if len(state) < len(self.state_names):  # (export passes it all: no traced batch size)
            state = self.initial_state(x.shape[0]) | state
        X = x[:, 0] * WNORM  # libDF's spectrum scale: its features and the network see this
        re, im = X[:, :BINS], X[:, BINS:]

        # features (libDF feat_erb / feat_cplx): band mean power in dB and the low bins, each against a running mean
        e = 10 * torch.log10((re ** 2 + im ** 2) @ self.erb_fb + 1e-10)
        s_erb = e * (1 - ALPHA) + (state["s_erb"] + ERB_INIT) * ALPHA
        mag = (re[:, :NB_DF] ** 2 + im[:, :NB_DF] ** 2).sqrt()
        s_spec = mag * (1 - ALPHA) + (state["s_spec"] + SPEC_INIT) * ALPHA
        norm = s_spec.clamp(min=1e-12).sqrt()  # long digital silence (a muted mic) decays the mean towards 0: no 0/0
        f_erb = torch.cat([state["f_erb"], ((e - s_erb) / 40)[:, None, None]], 2)  # [B, 1, 3, E]
        f_spec = torch.cat([state["f_spec"], torch.stack([re[:, :NB_DF] / norm, im[:, :NB_DF] / norm], 1)[:, :, None]], 2)

        # encoder (Encoder.forward for one frame)
        enc = self.enc
        e0 = enc.erb_conv0.frame(f_erb)
        e1 = enc.erb_conv1.frame(e0)
        e2 = enc.erb_conv2.frame(e1)
        e3 = enc.erb_conv3.frame(e2)
        c0 = enc.df_conv0.frame(f_spec)
        c1 = enc.df_conv1.frame(c0)
        emb = e3.permute(0, 2, 3, 1).flatten(2) + enc.df_fc_emb(c1.permute(0, 2, 3, 1).flatten(2))
        if spk is not None:
            emb = self.film(self.film_in, emb, spk)
        emb, h_enc = enc.emb_gru(emb, state["h_enc"])
        if spk is not None:
            emb = self.film(self.film_out, emb, spk)

        # ERB gains (ErbDecoder.forward)
        d = self.erb_dec
        y, h_erb = d.emb_gru(emb, state["h_erb"])
        y = y.unflatten(-1, (NB_ERB // 4, CH)).permute(0, 3, 1, 2)
        y = d.convt3.frame(d.conv3p.frame(e3) + y)
        y = d.convt2(d.conv2p.frame(e2) + y)
        y = d.convt1(d.conv1p.frame(e1) + y)
        m = d.gains(d.conv0p.frame(e0) + y)[:, 0]  # [B, E] (no time kernel: a frame works like a clip)

        # deep-filter coefficients (DfDecoder.forward), DF_ORDER complex taps over spectra t-2..t+2
        dd = self.df_dec
        c, h_df = dd.df_gru(emb, state["h_df"])
        c = c + dd.df_skip(emb)
        c0s = torch.cat([state["c0"], c0], 2)  # [B, CH, 5, NB_DF]
        coefs = dd.df_out(c)[:, 0].unflatten(-1, (NB_DF, DF_ORDER * 2)) + dd.df_convp.frame(c0s)[:, :, 0].transpose(1, 2)
        c_re, c_im = coefs[..., 0::2], coefs[..., 1::2]  # [B, NB_DF, DF_ORDER]: the reshape of DfOutputReshapeMF

        # apply to the frame 2 back: ERB gains on all bins, the deep filter replaces the low NB_DF
        specs = torch.cat([state["spec"], X[:, None]], 1)  # [B, 5, 2*BINS]: frames t-2 .. t+2 for output frame t
        o_re, o_im = specs[:, LOOK, :BINS], specs[:, LOOK, BINS:]
        g = m @ self.mask.erb_inv_fb
        h_re, h_im = specs[..., :NB_DF].transpose(1, 2), specs[..., BINS:BINS + NB_DF].transpose(1, 2)  # [B, F, 5]
        df_re = (h_re * c_re - h_im * c_im).sum(-1)
        df_im = (h_re * c_im + h_im * c_re).sum(-1)
        y_re = torch.cat([df_re, (g * o_re)[:, NB_DF:]], 1)
        y_im = torch.cat([df_im, (g * o_im)[:, NB_DF:]], 1)
        y = torch.cat([y_re, y_im], 1)[:, None] / WNORM

        return (y, s_erb - ERB_INIT, s_spec - SPEC_INIT, f_erb[:, :, 1:], f_spec[:, :, 1:], c0s[:, :, 1:], specs[:, 1:],
                h_enc, h_erb, h_df)

    def forward(self, wav, spk=None):
        """Training: noisy waveforms [B, N] -> enhanced [B, N], aligned with the input (the last frames see silence
        after the end); spk as in step(). The same computation as step() over a stream, run on whole clips: causal
        convs and GRUs over all frames, output frame k (for input frame k - LOOK) taken at k."""
        n = wav.shape[1]
        x = F.pad(wav, (HOP, HOP * (LOOK + 1) + (-n) % HOP))  # libDF's first frame: HOP zeros, then the first hop
        win = WINDOW.to(wav.device)
        X = torch.fft.rfft(x.unfold(1, FFT, HOP) * win) * WNORM  # [B, T, BINS]
        re, im = X.real, X.imag
        with torch.no_grad():  # features: running means of input levels, the same recurrences as step()
            e = 10 * torch.log10((re ** 2 + im ** 2) @ self.erb_fb + 1e-10)
            mag = (re[..., :NB_DF] ** 2 + im[..., :NB_DF] ** 2).sqrt()
            s_erb, s_spec = (ema(v, init.to(v.device)) for v, init in ((e, ERB_INIT), (mag, SPEC_INIT)))
            norm = s_spec.clamp(min=1e-12).sqrt()
            f_erb = ((e - s_erb) / 40)[:, None]  # [B, 1, T, E]
            f_spec = torch.stack([re[..., :NB_DF] / norm, im[..., :NB_DF] / norm], 1)  # [B, 2, T, NB_DF]

        enc = self.enc  # the modules' own forward: their ConstantPad2d is the causal time padding step() keeps as state
        e0 = enc.erb_conv0(f_erb)
        e1 = enc.erb_conv1(e0)
        e2 = enc.erb_conv2(e1)
        e3 = enc.erb_conv3(e2)
        c0 = enc.df_conv0(f_spec)
        c1 = enc.df_conv1(c0)
        emb = e3.permute(0, 2, 3, 1).flatten(2) + enc.df_fc_emb(c1.permute(0, 2, 3, 1).flatten(2))  # [B, T, EMB]
        if spk is not None:
            emb = self.film(self.film_in, emb, spk)
        emb, _ = enc.emb_gru(emb)
        if spk is not None:
            emb = self.film(self.film_out, emb, spk)

        d = self.erb_dec
        y, _ = d.emb_gru(emb)
        y = y.unflatten(-1, (NB_ERB // 4, CH)).permute(0, 3, 1, 2)  # [B, CH, T, E / 4]
        y = d.convt3(d.conv3p(e3) + y)
        y = d.convt2(d.conv2p(e2) + y)
        y = d.convt1(d.conv1p(e1) + y)
        m = d.gains(d.conv0p(e0) + y)  # [B, T, E]

        dd = self.df_dec
        c, _ = dd.df_gru(emb)
        c = c + dd.df_skip(emb)
        coefs = dd.df_out(c).unflatten(-1, (NB_DF, DF_ORDER * 2)) + dd.df_convp(c0).permute(0, 2, 3, 1)
        c_re, c_im = coefs[..., 0::2], coefs[..., 1::2]  # [B, T, NB_DF, DF_ORDER]

        T = X.shape[1]
        low = F.pad(torch.stack([re[..., :NB_DF], im[..., :NB_DF]], -1), (0, 0, 0, 0, DF_ORDER - 1, 0))
        h = low.unfold(1, DF_ORDER, 1)  # [B, T, NB_DF, 2, DF_ORDER]: tap j at k is frame k - 4 + j, as in step()
        df_re = (h[..., 0, :] * c_re - h[..., 1, :] * c_im).sum(-1)
        df_im = (h[..., 0, :] * c_im + h[..., 1, :] * c_re).sum(-1)
        g = m @ self.mask.erb_inv_fb
        o_re, o_im = (F.pad(a, (0, 0, LOOK, 0))[:, :T] for a in (re, im))  # frame k - LOOK at k
        Y = torch.complex(torch.cat([df_re, (g * o_re)[..., NB_DF:]], -1), torch.cat([df_im, (g * o_im)[..., NB_DF:]], -1))
        frames = torch.fft.irfft(Y[:, LOOK:] / WNORM, n=FFT) * win  # estimates of input frames 0 ..
        return (frames[:, 1:, :HOP] + frames[:, :-1, HOP:]).flatten(1)[:, :n]  # overlap-add; hop k-1 of the input


def ema(v, init):
    """libDF's running mean along time [B, T, F], from init [F]: s_t = (1 - ALPHA) v_t + ALPHA s_(t-1)."""
    s, out = init.expand(v.shape[0], -1), []
    for t in range(v.shape[1]):
        s = v[:, t] * (1 - ALPHA) + s * ALPHA
        out.append(s)
    return torch.stack(out, 1)


def load(path):
    """DFN3 from a DFN checkpoint (a plain state dict: the release, or a dfn-recipe run) or one of train.py's (an
    enrolled-voice one if its config has spk)."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = DFN3(spk=state.get("config", {}).get("spk", 0))
    state = state.get("model", state)
    state = {k: v for k, v in state.items() if k in model.state_dict()}
    missing = set(model.state_dict()) - set(state) - {"erb_dec.conv0_gain.weight", "erb_dec.conv0_gain.bias"}
    assert not missing, f"checkpoint lacks {sorted(missing)[:5]}"
    for k in ("erb_fb", "mask.erb_inv_fb"):  # ours are rebuilt from erb_widths(): they must agree with DFN's
        assert torch.allclose(state[k].float(), model.state_dict()[k]), k
    model.load_state_dict(state, strict=False)  # a DFN checkpoint has no boost head: it keeps its 0 start
    return model.eval()


def stream(model, wav, spk=None):
    """The app's loop, as in model.stream(): 480 samples in, 480 out, lagging 480 * (1 + LOOK) samples."""
    buf, tail, state, out = torch.zeros(FFT), torch.zeros(HOP), {}, []
    with torch.no_grad():
        for i in range(0, len(wav) - HOP + 1, HOP):
            buf = torch.cat([buf[HOP:], wav[i:i + HOP]])
            X = torch.fft.rfft(buf * WINDOW)
            y, *s = model.step(torch.cat([X.real, X.imag])[None, None], spk, **state)
            state = dict(zip(model.state_names, s))
            frame = torch.fft.irfft(torch.complex(*y[0, 0].chunk(2)), n=FFT) * WINDOW
            out.append(tail + frame[:HOP])
            tail = frame[HOP:]
    return torch.cat(out)


if __name__ == "__main__" and sys.argv[1] == "check":
    import soundfile as sf
    ckpt, inp, ref = sys.argv[2:5]
    x, sr = sf.read(inp, dtype="float32")
    r, _ = sf.read(ref, dtype="float32")
    assert sr == SR and x.ndim == 1
    model = load(ckpt)
    assert torch.isfinite(stream(model, torch.zeros(SR))).all(), "digital silence gives NaN"
    y = stream(model, torch.from_numpy(x)).numpy()[HOP * (1 + LOOK):]
    n = min(len(y), len(r)) - SR // 10
    a, b = y[SR // 2:n], r[SR // 2:n]  # skip the start: offline DFN zero-pads its first frames, the stream does not
    snr = 10 * np.log10((b ** 2).sum() / ((a - b) ** 2).sum())
    print(f"streaming DFN3 vs DFN's enhance(): {snr:.1f} dB SNR over {len(a) / SR:.1f} s")
    assert snr > 30, "streaming port does not match DFN"

if __name__ == "__main__" and sys.argv[1] == "selftest":
    # forward() (training) must compute what step() does in the app, with and without a boost; both on noisy speech-
    # like input with a level change, so the running means and the boost head are exercised
    torch.manual_seed(0)
    model = load(sys.argv[2])
    t = torch.arange(3 * SR) / SR
    wav = (0.1 * torch.sin(2 * math.pi * 220 * t) * (1 + torch.sin(2 * math.pi * 3 * t)) * (t > 1)
           + 0.01 * torch.randn(3 * SR))
    lag = HOP * (1 + LOOK)
    for boost in (False, True):
        if boost:
            nn.init.normal_(model.erb_dec.conv0_gain.weight, std=0.05)
        with torch.no_grad():
            a, b = model(wav[None])[0, :-lag], stream(model, wav)[lag:]
        snr = 10 * torch.log10(b.pow(2).sum() / (a - b).pow(2).sum())
        print(f"forward() vs stream(){' with boost' if boost else ''}: {snr:.1f} dB SNR")
        assert snr > 60, "the training path differs from the app's"
    x = torch.randn(2, SR // 2 + 123)
    assert model(x).shape == x.shape  # any length
    # the enrolled voice: starts as the plain model; once trained, forward() and step() must still agree
    enrolled = DFN3(spk=512)
    enrolled.load_state_dict(model.state_dict(), strict=False)
    enrolled.eval()
    spk = F.normalize(torch.randn(1, 512), dim=-1)
    with torch.no_grad():
        assert torch.equal(enrolled(wav[None], spk), model(wav[None])), "a new enrolled model is not the plain one"
        for f in (enrolled.film_in, enrolled.film_out):
            nn.init.normal_(f[2].weight, std=0.05)
        a, b = enrolled(wav[None], spk)[0, :-lag], stream(enrolled, wav, spk)[lag:]
        snr = 10 * torch.log10(b.pow(2).sum() / (a - b).pow(2).sum())
        print(f"forward() vs stream() with an enrolled voice: {snr:.1f} dB SNR")
        assert snr > 60, "the enrolled training path differs from the app's"
        assert torch.equal(enrolled(wav[None], torch.zeros(1, 512)), model(wav[None])), "no voice is not the plain model"
