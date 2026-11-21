"""StudioNet: causal full-band (48 kHz) dual-path time-frequency speech restoration model for Core ML / Apple Silicon.

Drop-in compatible with mac/Sources/CleanMyMic/Enhancer.swift and train/export.py:
- 48 kHz mono, 960-sample (20 ms) sqrt-Hann window, 480-sample (10 ms) hop, 481 bins.
- Compressed complex STFT input/output (c = 0.3), explicit named streaming state, exact forward() == stream() parity.
- Hybrid output head: 5-tap complex deep filter (0-4.8 kHz) + unbounded-boost complex mask + additive complex
  synthesis residual across all 481 bins (0-24 kHz), enabling dereverberation, mic EQ restoration, and bandwidth
  extension (8->24 kHz) while initializing to clean passthrough and producing exact silence on digital silence.
"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from model import (
    BINS,
    C,
    DF_BINS,
    DF_ORDER,
    HIST,
    HOP,
    SR,
    WIN,
    _power,
    compress,
    decompress,
)

EMA_ALPHA = 0.99  # ~1 s running mean of per-bin log magnitude (tracks room/mic coloration and noise floor)
EMA_INIT = (
    -8.0
)  # stored offset so zero-initialized Mac app state starts at a quiet floor


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels C for [B, C, T, F] tensors; identical in batch and 1-frame streaming mode."""

    def __init__(self, ch):
        super().__init__()
        self.norm = nn.LayerNorm(ch)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DualPathTFBlock(nn.Module):
    """One causal time-frequency block at [B, C, T, F]:
    1. Full-band Frequency Self-Attention + FFN (connects 0-8 kHz speech harmonics to 8-24 kHz in one hop; zero state).
    2. Causal Per-Band Time GRU (tracks reverb decay & harmonic continuity per band; 1 state tensor [1, B*F, C]).
    """

    def __init__(self, ch, n_freq, heads=4):
        super().__init__()
        self.ch, self.n_freq, self.heads = ch, n_freq, heads
        self.head_dim = ch // heads
        assert ch % heads == 0
        # Frequency self-attention (explicit Q/K/V matmuls for clean Core ML / ANE tracing)
        self.fnorm1 = nn.LayerNorm(ch)
        self.qkv = nn.Linear(ch, 3 * ch)
        self.proj = nn.Linear(ch, ch)
        self.fnorm2 = nn.LayerNorm(ch)
        self.ffn = nn.Sequential(
            nn.Linear(ch, ch * 2), nn.SiLU(), nn.Linear(ch * 2, ch)
        )
        # Causal per-band temporal GRU
        self.tnorm = nn.LayerNorm(ch)
        self.tgru = nn.GRU(ch, ch, batch_first=True)
        self.tproj = nn.Linear(ch, ch)

    def forward(self, x, h=None):
        # x: [B, C, T, F], h: [1, B*F, C] or None; zero dynamic .shape reads for clean Core ML tracing
        # 1) Frequency axis: [B, T, F, C] (max tensor rank 5 for Core ML)
        xf = x.permute(0, 2, 3, 1)
        q, k, v = self.qkv(self.fnorm1(xf)).chunk(3, dim=-1)
        q = q.unflatten(-1, (self.heads, self.head_dim)).transpose(-3, -2)
        k = k.unflatten(-1, (self.heads, self.head_dim)).transpose(-3, -2)
        v = v.unflatten(-1, (self.heads, self.head_dim)).transpose(-3, -2)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * (self.head_dim**-0.5), dim=-1)
        xf = xf + self.proj((attn @ v).transpose(-3, -2).flatten(-2))
        xf = xf + self.ffn(self.fnorm2(xf))
        # 2) Time axis (causal per band): [B*F, T, C]
        xt = xf.permute(0, 2, 1, 3).flatten(0, 1)
        zt, h_out = self.tgru(self.tnorm(xt), h)
        xt = xt + self.tproj(zt)
        out = xt.unflatten(0, (-1, self.n_freq)).permute(0, 3, 2, 1)
        return out, h_out


class StudioNet(nn.Module):
    """Causal 48 kHz dual-path studio speech restoration network (Stage-1 regression backbone).

    Optionally accepts a fixed speaker embedding `spk` [B, spk_dim] via zero-init FiLM conditioning:
    - When `spk_dim == 0` or `spk` is all zeros, runs identically in zero-enrollment mode.
    - When `spk` is populated (e.g., 5-second enrollment), locks onto the primary speaker's voice.
    """

    def __init__(
        self,
        ch=(32, 48, 64, 80),
        n_blocks=4,
        heads=4,
        global_hidden=192,
        look=2,
        past=2,
        df=True,
        spk_dim=0,
    ):
        super().__init__()
        assert 0 <= look and 0 <= past and look + past <= HIST
        self.ch = tuple(ch)
        self.n_blocks = n_blocks
        self.heads = heads
        self.global_hidden = global_hidden
        self.look, self.past, self.df, self.spk_dim = look, past, df, spk_dim

        in_ch = (
            2 * (look + 1 + past) + 2
        )  # stacked (re, im) frames + current log-mag + (log-mag - ema)
        # Frequency-only encoder: 481 -> 241 -> 121 -> 61
        self.enc = nn.ModuleList()
        self.enc_norm = nn.ModuleList()
        prev = in_ch
        for i, c in enumerate(self.ch):
            stride = 1 if i == 0 else 2
            self.enc.append(
                nn.Conv2d(prev, c, (1, 5), stride=(1, stride), padding=(0, 2))
            )
            self.enc_norm.append(ChannelLayerNorm(c))
            prev = c

        self.n_freq = (BINS - 1) // (2 ** (len(self.ch) - 1)) + 1  # 61 for 4 levels
        mid_c = self.ch[-1]

        # Learnable frequency-band positional embedding so self-attention knows 0-4 kHz vs 8-24 kHz bands
        self.freq_pos = nn.Parameter(torch.zeros(1, mid_c, 1, self.n_freq))
        nn.init.normal_(self.freq_pos, std=0.02)

        # Global utterance-level context GRU (tracks primary speaker vs pauses & background voices over 40s streams)
        self.g_in = nn.Linear(mid_c * 2, global_hidden)
        self.g_gru = nn.GRU(
            global_hidden, global_hidden, num_layers=2, batch_first=True
        )
        self.g_film = nn.Linear(global_hidden, mid_c * 2)
        nn.init.zeros_(self.g_film.weight)
        nn.init.zeros_(self.g_film.bias)

        # Optional zero-init speaker FiLM conditioning (exact identity when spk == 0 -> zero-enrollment default)
        if spk_dim > 0:
            self.spk_g = nn.Linear(spk_dim, global_hidden, bias=False)
            self.spk_film = nn.Linear(spk_dim, mid_c * 2, bias=False)
            nn.init.zeros_(self.spk_g.weight)
            nn.init.zeros_(self.spk_film.weight)
        else:
            self.spk_g = None
            self.spk_film = None

        # Dual-path time-frequency restoration core
        self.blocks = nn.ModuleList(
            DualPathTFBlock(mid_c, self.n_freq, heads=heads) for _ in range(n_blocks)
        )

        # Symmetric frequency decoder with skip connections: 61 -> 121 -> 241 -> 481
        self.dec = nn.ModuleList()
        self.dec_norm = nn.ModuleList()
        for a, b in zip(self.ch[:0:-1], self.ch[-2::-1]):
            self.dec.append(
                nn.ConvTranspose2d(a, b, (1, 5), stride=(1, 2), padding=(0, 2))
            )
            self.dec_norm.append(ChannelLayerNorm(b))

        # Hybrid output head at full 481-bin resolution:
        # - complex mask (2 ch: m_re, m_im) applied to centered input frame X_t (can attenuate or boost +20 dB)
        # - additive complex synthesis residual (2 ch: r_re, r_im) for high-band synthesis & dereverb
        # - optional 5-tap complex deep filter (2 * DF_ORDER ch) for low bins 0-4.8 kHz
        out_ch = 4 + (2 * DF_ORDER if df else 0)
        self.head = nn.Sequential(
            nn.Conv2d(self.ch[0], self.ch[0], (1, 5), padding=(0, 2)),
            ChannelLayerNorm(self.ch[0]),
            nn.SiLU(),
            nn.Conv2d(self.ch[0], out_ch, (1, 3), padding=(0, 1)),
        )
        # Initialize as clean passthrough (mask = 1+0j, residual = 0, DF centered tap = 1+0j)
        last = self.head[-1]
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.zero_()
            last.bias[0] = math.atanh(1.0 / 3.0)  # m_re = 3 * tanh(bias) = 1.0
            if df:
                last.bias[4 + look] = math.atanh(
                    1.0 / 2.0
                )  # centered DF tap = 2 * tanh(bias) = 1.0

        self.register_buffer("window", torch.hann_window(WIN).sqrt())

    @property
    def state_names(self):
        names = ["buf", "ema", "hg"] + [f"hb{i}" for i in range(self.n_blocks)]
        if self.spk_dim > 0:
            names.append("spk")
        return names

    def _ema(self, log_mag, ema_state):
        """Per-bin causal EMA [B, T, BINS]; ema_state [B, BINS] stores offset from EMA_INIT."""
        with torch.no_grad():
            if ema_state is None:
                ema_state = log_mag.new_zeros(log_mag.shape[0], BINS)
            s = ema_state + EMA_INIT
            if log_mag.size(1) == 1:
                s = log_mag[:, 0] * (1.0 - EMA_ALPHA) + s * EMA_ALPHA
                return s[:, None], s - EMA_INIT
            outs = []
            for t in range(log_mag.size(1)):
                s = log_mag[:, t] * (1.0 - EMA_ALPHA) + s * EMA_ALPHA
                outs.append(s)
            return torch.stack(outs, 1), s - EMA_INIT

    def _step_core(self, x, buf=None, ema=None, hg=None, spk=None, **hb_states):
        """Core streaming/batch step returning (y, g_out, x_cur, new_states_tuple)."""
        if buf is None:
            buf = x.new_zeros(x.shape[0], HIST, 2 * BINS)
        hist = torch.cat([buf, x], 1)
        back = lambda j: hist[:, HIST - j : -j] if j else hist[:, HIST:]
        frames = list(range(self.look, -1, -1)) + list(
            range(self.look + 1, self.look + 1 + self.past)
        )
        chans = [p for j in frames for p in back(j).chunk(2, -1)]

        # Centered frame (output target frame t - look) + newest frame for running floor
        x_cur = back(self.look)
        x_re, x_im = x_cur.chunk(2, -1)
        mag_c = (x_re * x_re + x_im * x_im + 1e-12).sqrt()
        log_mag = (mag_c + 1e-4).log()
        ema_seq, ema_out = self._ema(log_mag, ema)
        chans.append(log_mag * 0.25)
        chans.append((log_mag - ema_seq) * 0.5)

        a = torch.stack(chans, 1)  # [B, in_ch, T, BINS]
        skips = []
        for conv, norm in zip(self.enc, self.enc_norm):
            a = F.silu(norm(conv(a)))
            skips.append(a)

        # Global utterance context across frequency (mean + max pooling) + optional speaker embedding
        g_feat = torch.cat([a.mean(-1), a.amax(-1)], 1).transpose(
            1, 2
        )  # [B, T, 2*mid_c]
        g_pre = self.g_in(g_feat)
        spk_out = None
        if self.spk_dim > 0:
            if spk is None:
                spk = x.new_zeros(x.shape[0], self.spk_dim)
            # Distinct graph node so Core ML FP32 NameSanitizer allows separate 'spk' input and 'spk_out' output
            spk_out = spk + 0.0
            g_pre = g_pre + self.spk_g(spk).unsqueeze(1)
        g_out, hg_out = self.g_gru(F.silu(g_pre), hg)
        gamma, beta = self.g_film(g_out).transpose(1, 2).unsqueeze(-1).chunk(2, 1)
        if self.spk_dim > 0:
            sg, sb = self.spk_film(spk).unsqueeze(-1).unsqueeze(-1).chunk(2, 1)
            gamma, beta = gamma + sg, beta + sb
        a = a * (1.0 + gamma) + beta + self.freq_pos

        # Dual-path time-frequency blocks (gradient checkpointing only for long >10s training streams to cap VRAM)
        use_ckpt = self.training and a.requires_grad and a.size(2) > 1000
        hb_outs = []
        for i, block in enumerate(self.blocks):
            hi = hb_states.get(f"hb{i}", None)
            if use_ckpt and hi is None:
                a = grad_checkpoint(
                    lambda z, b=block: b(z, None)[0], a, use_reentrant=False
                )
                hi_out = None
            else:
                a, hi_out = block(a, hi)
            hb_outs.append(hi_out)

        # Decode back to 481 bins
        for deconv, norm, skip in zip(self.dec, self.dec_norm, skips[-2::-1]):
            a = F.silu(norm(deconv(a))) + skip

        o = self.head(a)  # [B, out_ch, T, BINS]
        m_re = 3.0 * torch.tanh(o[:, 0])
        m_im = 3.0 * torch.tanh(o[:, 1])

        # Scale-equivariant bounded synthesis residual: scales automatically from -50 dBFS to -12 dBFS,
        # exact 0 on digital silence, bounded smoothly for GAN stability.
        frame_energy = (x_re * x_re + x_im * x_im).mean(-1, keepdim=True).sqrt()
        r_re = frame_energy * 4.0 * torch.tanh(o[:, 2] * 0.25)
        r_im = frame_energy * 4.0 * torch.tanh(o[:, 3] * 0.25)

        # Hybrid complex output in compressed domain (mask * X + scale-equivariant synthesis residual)
        y_re = m_re * x_re - m_im * x_im + r_re
        y_im = m_re * x_im + m_im * x_re + r_im

        if self.df:
            # 5-tap complex deep filter in FP32 linear spectrum on 0-4.8 kHz + additive compressed residual
            with torch.autocast(x.device.type, enabled=False):
                h_re, h_im = _power(
                    hist[..., :DF_BINS].float(),
                    hist[..., BINS : BINS + DF_BINS].float(),
                    1.0 / C,
                )
                c_df = 2.0 * torch.tanh(o[:, 4:, :, :DF_BINS].float())
                d_re = d_im = 0.0
                for j in range(DF_ORDER):
                    p_re, p_im = (
                        (h_re[:, HIST:], h_im[:, HIST:])
                        if j == 0
                        else (h_re[:, HIST - j : -j], h_im[:, HIST - j : -j])
                    )
                    cj_re, cj_im = c_df[:, j], c_df[:, DF_ORDER + j]
                    d_re = d_re + cj_re * p_re - cj_im * p_im
                    d_im = d_im + cj_re * p_im + cj_im * p_re
                df_c_re, df_c_im = _power(d_re, d_im, C)
                low_re = df_c_re + r_re[..., :DF_BINS].float()
                low_im = df_c_im + r_im[..., :DF_BINS].float()
                y_re = torch.cat([low_re, y_re[..., DF_BINS:].float()], -1)
                y_im = torch.cat([low_im, y_im[..., DF_BINS:].float()], -1)

        y = torch.cat([y_re, y_im], -1)
        states = (hist[:, -HIST:], ema_out, hg_out, *hb_outs) + (
            (spk_out,) if self.spk_dim > 0 else ()
        )
        return y, g_out, x_cur, states

    def step(self, x, buf=None, ema=None, hg=None, spk=None, **hb_states):
        """x: compressed noisy STFT frames [B, T, 2*BINS]. Returns (y, *states) matching state_names."""
        y, _, _, states = self._step_core(
            x, buf=buf, ema=ema, hg=hg, spk=spk, **hb_states
        )
        return (y, *states)

    def forward(self, wav, spk=None):
        with torch.autocast(wav.device.type, enabled=False):
            wav_f = wav.float()
            X = compress(
                torch.stft(
                    wav_f,
                    WIN,
                    HOP,
                    window=self.window.float(),
                    pad_mode="constant",
                    return_complex=True,
                )
            )
            x = F.pad(
                torch.cat([X.real, X.imag], 1).transpose(1, 2), (0, 0, 0, self.look)
            )
        y = self.step(x, spk=spk)[0][:, self.look :]
        with torch.autocast(wav.device.type, enabled=False):
            Y = decompress(torch.complex(*y.float().transpose(1, 2).chunk(2, 1)))
            return torch.istft(
                Y, WIN, HOP, window=self.window.float(), length=wav.shape[-1]
            )


class StudioRefiner(nn.Module):
    """Causal Stage-2 residual GAN refiner (RE-USE 2026 optimal-transport two-stage pattern).

    Sees Stage-1 compressed output Y1, raw noisy frame X_t (recovers over-suppressed consonants/breaths), and
    Stage-1's global context g_out via zero-init FiLM. Predicts a zero-init bounded complex refinement +
    scale-equivariant additive synthesis residual (exact identity passthrough of Stage 1 at step 0!).
    """

    def __init__(self, ch=(32, 64, 96), n_blocks=3, heads=4, ctx_dim=384):
        super().__init__()
        self.ch = tuple(ch)
        self.n_blocks = n_blocks
        self.heads = heads
        self.ctx_dim = ctx_dim
        self.n_freq = (BINS - 1) // 8 + 1  # 481 -> 241 -> 121 -> 61

        self.enc0 = nn.Sequential(
            nn.Conv2d(4, ch[0], (1, 5), stride=(1, 1), padding=(0, 2)),
            ChannelLayerNorm(ch[0]),
            nn.SiLU(),
        )
        self.enc1 = nn.Sequential(
            nn.Conv2d(ch[0], ch[1], (1, 5), stride=(1, 2), padding=(0, 2)),
            ChannelLayerNorm(ch[1]),
            nn.SiLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(ch[1], ch[2], (1, 5), stride=(1, 2), padding=(0, 2)),
            ChannelLayerNorm(ch[2]),
            nn.SiLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(ch[2], ch[2], (1, 5), stride=(1, 2), padding=(0, 2)),
            ChannelLayerNorm(ch[2]),
            nn.SiLU(),
        )

        self.freq_pos = nn.Parameter(torch.zeros(1, ch[2], 1, self.n_freq))
        nn.init.normal_(self.freq_pos, std=0.02)

        self.ctx_film = nn.Linear(ctx_dim, ch[2] * 2)
        nn.init.zeros_(self.ctx_film.weight)
        nn.init.zeros_(self.ctx_film.bias)

        self.blocks = nn.ModuleList(
            DualPathTFBlock(ch[2], self.n_freq, heads=heads) for _ in range(n_blocks)
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(ch[2], ch[2], (1, 5), stride=(1, 2), padding=(0, 2)),
            ChannelLayerNorm(ch[2]),
            nn.SiLU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(ch[2], ch[1], (1, 5), stride=(1, 2), padding=(0, 2)),
            ChannelLayerNorm(ch[1]),
            nn.SiLU(),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(ch[1], ch[0], (1, 5), stride=(1, 2), padding=(0, 2)),
            ChannelLayerNorm(ch[0]),
            nn.SiLU(),
        )

        self.out = nn.Sequential(
            nn.Conv2d(ch[0], ch[0], (1, 5), padding=(0, 2)),
            ChannelLayerNorm(ch[0]),
            nn.SiLU(),
            nn.Conv2d(ch[0], 4, (1, 3), padding=(0, 1)),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, y1, x_cur, g_out, hr_states=None):
        """y1, x_cur: [B, T, 2*BINS] compressed STFT; g_out: [B, T, ctx_dim]. Returns (y_refined, hr_outs)."""
        y1_re, y1_im = y1.chunk(2, -1)
        x_re, x_im = x_cur.chunk(2, -1)
        inp = torch.stack([y1_re, y1_im, x_re, x_im], 1)

        e0 = self.enc0(inp)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        h = self.enc3(e2)

        gamma, beta = self.ctx_film(g_out).transpose(1, 2).unsqueeze(-1).chunk(2, 1)
        h = h * (1.0 + gamma) + beta + self.freq_pos

        use_ckpt = self.training and h.requires_grad and h.size(2) > 1000
        hr_outs = []
        for i, blk in enumerate(self.blocks):
            hi = hr_states.get(f"hr{i}", None) if hr_states else None
            if use_ckpt and hi is None:
                h = grad_checkpoint(
                    lambda z, b=blk: b(z, None)[0], h, use_reentrant=False
                )
                hi_out = None
            else:
                h, hi_out = blk(h, hi)
            hr_outs.append(hi_out)

        d = self.dec3(h) + e2
        d = self.dec2(d) + e1
        d = self.dec1(d) + e0
        o = self.out(d)

        with torch.autocast(y1.device.type, enabled=False):
            o = o.float()
            y1_re_f, y1_im_f = y1_re.float(), y1_im.float()
            dm_re = 2.0 * torch.tanh(o[:, 0] * 0.5)
            dm_im = 2.0 * torch.tanh(o[:, 1] * 0.5)
            y1_energy = (
                (y1_re_f * y1_re_f + y1_im_f * y1_im_f).mean(-1, keepdim=True).sqrt()
            )
            r_re = y1_energy * 2.5 * torch.tanh(o[:, 2] * 0.4)
            r_im = y1_energy * 2.5 * torch.tanh(o[:, 3] * 0.4)
            out_re = (1.0 + dm_re) * y1_re_f - dm_im * y1_im_f + r_re
            out_im = (1.0 + dm_re) * y1_im_f + dm_im * y1_re_f + r_im
        return torch.cat([out_re, out_im], -1), hr_outs


class StudioNetV2(nn.Module):
    """Two-stage causal 48 kHz studio restoration model: Stage-1 Regression + Stage-2 Residual GAN Refiner.

    Train Stage 1 to convergence for maximum speech fidelity, then freeze Stage 1 (`freeze_stage1()`) and train
    the zero-init Stage-2 `StudioRefiner` with GAN + perceptual losses to restore high-band air & studio crispness
    without degrading Stage-1 word accuracy or speaker fidelity. Both stages trace into one Core ML .mlpackage.
    """

    def __init__(self, s1_cfg=None, ref_ch=(32, 64, 96), ref_blocks=3, ref_heads=4):
        super().__init__()
        s1_cfg = (
            dict(
                ch=(48, 80, 128, 160),
                n_blocks=8,
                heads=4,
                global_hidden=384,
                look=2,
                past=2,
                df=True,
                spk_dim=0,
            )
            if s1_cfg is None
            else dict(s1_cfg)
        )
        self.s1_cfg = s1_cfg
        self.ref_ch = tuple(ref_ch)
        self.ref_blocks = ref_blocks
        self.ref_heads = ref_heads
        self.stage1 = StudioNet(**s1_cfg)
        self.refiner = StudioRefiner(
            ch=self.ref_ch,
            n_blocks=ref_blocks,
            heads=ref_heads,
            ctx_dim=self.stage1.global_hidden,
        )
        self.look = self.stage1.look
        self.spk_dim = self.stage1.spk_dim
        self.register_buffer("window", self.stage1.window.clone())

    def freeze_stage1(self):
        """Freeze Stage-1 weights so Stage-2 GAN gradients cannot corrupt fidelity or hold-pause suppression."""
        self.stage1.eval()
        for p in self.stage1.parameters():
            p.requires_grad_(False)

    @property
    def state_names(self):
        return list(self.stage1.state_names) + [
            f"hr{i}" for i in range(self.ref_blocks)
        ]

    def step(self, x, **state):
        s1_names = self.stage1.state_names
        s1_in = {k: state[k] for k in s1_names if k in state}
        y1, g_out, x_cur, s1_states = self.stage1._step_core(x, **s1_in)
        hr_in = {
            f"hr{i}": state[f"hr{i}"]
            for i in range(self.ref_blocks)
            if f"hr{i}" in state
        }
        y, hr_outs = self.refiner(y1, x_cur, g_out, hr_states=hr_in)
        return (y, *s1_states, *hr_outs)

    def forward(self, wav, spk=None):
        with torch.autocast(wav.device.type, enabled=False):
            wav_f = wav.float()
            X = compress(
                torch.stft(
                    wav_f,
                    WIN,
                    HOP,
                    window=self.window.float(),
                    pad_mode="constant",
                    return_complex=True,
                )
            )
            x = F.pad(
                torch.cat([X.real, X.imag], 1).transpose(1, 2), (0, 0, 0, self.look)
            )
        s1_frozen = not any(p.requires_grad for p in self.stage1.parameters())
        if s1_frozen:
            with torch.no_grad():
                y1, g_out, x_cur, _ = self.stage1._step_core(x, spk=spk)
            y, _ = self.refiner(y1.detach(), x_cur.detach(), g_out.detach())
        else:
            y1, g_out, x_cur, _ = self.stage1._step_core(x, spk=spk)
            y, _ = self.refiner(y1, x_cur, g_out)
        y = y[:, self.look :]
        with torch.autocast(wav.device.type, enabled=False):
            Y = decompress(torch.complex(*y.float().transpose(1, 2).chunk(2, 1)))
            return torch.istft(
                Y, WIN, HOP, window=self.window.float(), length=wav.shape[-1]
            )


STUDIO_PRESETS = {
    # Compact M1 / low-power target (can be trained directly or distilled from Pro/Max)
    "lite": dict(
        s1=dict(
            ch=(32, 64, 96, 128),
            n_blocks=6,
            heads=4,
            global_hidden=256,
            look=2,
            past=2,
            df=True,
        ),
        ref=dict(ref_ch=(24, 48, 64), ref_blocks=2, ref_heads=4),
    ),
    # Legacy m/l aliases preserved for existing checkpoints & scripts
    "m": dict(
        s1=dict(
            ch=(32, 48, 64, 96),
            n_blocks=6,
            heads=4,
            global_hidden=256,
            look=2,
            past=2,
            df=True,
        ),
        ref=dict(ref_ch=(24, 48, 64), ref_blocks=2, ref_heads=4),
    ),
    "l": dict(
        s1=dict(
            ch=(32, 64, 96, 128),
            n_blocks=6,
            heads=4,
            global_hidden=256,
            look=2,
            past=2,
            df=True,
        ),
        ref=dict(ref_ch=(24, 48, 80), ref_blocks=2, ref_heads=4),
    ),
    # Recommended M3/M4 Pro + M5 Pro studio flagship (~6.5M two-stage, ~13 MB mixed, ~1.7 ms/hop on M5 Pro)
    "pro": dict(
        s1=dict(
            ch=(48, 80, 128, 160),
            n_blocks=8,
            heads=4,
            global_hidden=384,
            look=2,
            past=2,
            df=True,
        ),
        ref=dict(ref_ch=(32, 64, 96), ref_blocks=3, ref_heads=4),
    ),
    # High-capacity M4/M5 Pro model / distillation teacher (~9.2M two-stage, ~19 MB mixed, ~2.2 ms/hop on M5 Pro)
    "max": dict(
        s1=dict(
            ch=(48, 96, 144, 192),
            n_blocks=10,
            heads=6,
            global_hidden=384,
            look=2,
            past=2,
            df=True,
        ),
        ref=dict(ref_ch=(32, 64, 96), ref_blocks=3, ref_heads=4),
    ),
}


def stream(model, wav, spk=None):
    """Exact mirror of mac/Sources/CleanMyMic/Enhancer.swift."""
    buf, tail, state, out = torch.zeros(WIN), torch.zeros(HOP), {}, []
    if spk is not None and getattr(model, "spk_dim", 0) > 0:
        state["spk"] = spk if spk.ndim == 2 else spk[None]
    with torch.no_grad():
        for i in range(0, len(wav) - HOP + 1, HOP):
            buf = torch.cat([buf[HOP:], wav[i : i + HOP]])
            X = compress(torch.fft.rfft(buf * model.window))
            y, *s = model.step(torch.cat([X.real, X.imag])[None, None], **state)
            state = dict(zip(model.state_names, s))
            frame = (
                torch.fft.irfft(decompress(torch.complex(*y[0, 0].chunk(2))), n=WIN)
                * model.window
            )
            out.append(tail + frame[:HOP])
            tail = frame[HOP:]
    return torch.cat(out)


def load(path, map_location="cpu"):
    state = torch.load(path, map_location=map_location, weights_only=False)
    raw_cfg = dict(state.get("config", {}))
    arch = raw_cfg.pop("arch", "studio")
    if arch == "studio_v2":
        model = StudioNetV2(**raw_cfg)
    else:
        model = StudioNet(**raw_cfg)
    model.load_state_dict(state["model"])
    return model.eval(), state


if __name__ == "__main__":
    torch.manual_seed(0)
    t = torch.arange(3 * SR) / SR
    wav = torch.randn(3 * SR) * 0.1 * (1.1 + torch.sin(2 * math.pi * 0.7 * t))

    # 1. Stage-1 StudioNet (with optional speaker FiLM)
    model = StudioNet(spk_dim=64).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    spk_vec = F.normalize(torch.randn(1, 64), dim=-1)
    with torch.no_grad():
        pt = model(wav[None])[0]
        p_err = (pt - wav)[WIN:-WIN].abs().max().item()
        assert p_err < 1e-3, f"untrained model not passthrough: {p_err}"
        sil = model(torch.zeros(1, SR))[0]
        assert sil.abs().max().item() == 0.0, (
            f"silence not zero: {sil.abs().max().item()}"
        )
        model.head[-1].weight.normal_(0, 0.02)
        model.head[-1].bias.normal_(0, 0.02)
        model.spk_film.weight.normal_(0, 0.02)
        model.spk_g.weight.normal_(0, 0.02)
        offline, streamed = (
            model(wav[None], spk=spk_vec)[0],
            stream(model, wav, spk=spk_vec),
        )
        sil_after = model(torch.zeros(1, SR), spk=spk_vec)[0]
        assert sil_after.abs().max().item() == 0.0, (
            f"trained silence not zero: {sil_after.abs().max().item()}"
        )
    lag = HOP * (1 + model.look)
    err = (streamed[lag:] - offline[: len(streamed) - lag]).abs().max().item()
    snr = 10 * math.log10(
        (
            offline[: len(streamed) - lag].pow(2).sum()
            / (streamed[lag:] - offline[: len(streamed) - lag])
            .pow(2)
            .sum()
            .clamp(min=1e-20)
        ).item()
    )
    assert err < 1e-4, f"streaming != offline: max diff {err}"
    print(
        f"ok StudioNet: {n_params:.2f}M params, passthrough err {p_err:.1e}, stream vs offline {snr:.1f} dB SNR (max diff {err:.1e}), silence=0"
    )

    # 2. Two-stage StudioNetV2 (step 0 == Stage 1, perturbed stream == offline, silence == 0)
    s1_cfg = dict(
        ch=(16, 24, 32, 48),
        n_blocks=2,
        heads=4,
        global_hidden=64,
        look=2,
        past=2,
        df=True,
        spk_dim=64,
    )
    v2 = StudioNetV2(
        s1_cfg=s1_cfg, ref_ch=(16, 24, 32), ref_blocks=2, ref_heads=4
    ).eval()
    with torch.no_grad():
        v2.stage1.head[-1].weight.normal_(0, 0.02)
        y_s1 = v2.stage1(wav[None], spk=spk_vec)[0]
        y_v2_init = v2(wav[None], spk=spk_vec)[0]
        id_err = (y_v2_init - y_s1).abs().max().item()
        assert id_err < 1e-5, f"StudioNetV2 init != Stage1: {id_err}"
        v2.refiner.out[-1].weight.normal_(0, 0.02)
        v2.refiner.out[-1].bias.normal_(0, 0.02)
        v2_off, v2_str = v2(wav[None], spk=spk_vec)[0], stream(v2, wav, spk=spk_vec)
        v2_sil = v2(torch.zeros(1, SR), spk=spk_vec)[0]
        assert v2_sil.abs().max().item() == 0.0, (
            f"StudioNetV2 silence not zero: {v2_sil.abs().max().item()}"
        )
    v2_err = (v2_str[lag:] - v2_off[: len(v2_str) - lag]).abs().max().item()
    assert v2_err < 1e-4, f"StudioNetV2 streaming != offline: {v2_err}"
    print(
        f"ok StudioNetV2: init==S1 err {id_err:.1e}, stream vs offline max diff {v2_err:.1e}, silence=0"
    )
