"""DFN3Studio: Unfrozen DeepFilterNet3 (CleanMyMic Base) + zero-init Full-Band Frequency-Attention Restoration Head.

Why this architecture:
- Starts 100% bit-identical to CleanMyMic Base (hold-10000.pt, 150k+ pretrained steps, 0.21 ms/frame) at step 0.
- Unfreezes all of DFN3 (encoder, GRUs, boost head conv0_gain, deep-filter decoder) so the backbone learns
  time-aligned dry anechoic dereverberation, mic de-coloring, and WavLM word preservation instead of fighting a frozen suppressor.
- Adds a lightweight Core ML-native RestoreHead (481 -> 121 -> 61 -> Frequency Self-Attention -> 121 -> 481) that sees
  BOTH the noisy input spectrum AND DFN3's estimate plus DFN3's encoder embedding, predicting a scale-equivariant
  bounded complex residual across all 481 bins (0-24 kHz) — enabling 8->24 kHz bandwidth extension, harmonic
  restoration, and fine-grained dereverberation with 0 extra lookahead (same 40 ms window+lookahead as Base).
- Drop-in compatible with mac/Sources/CleanMyMic/Enhancer.swift (vorbis window, compress=1, look=2).
"""
import math
import torch
import torch.nn.functional as F
from torch import nn

import dfn3
from dfn3 import BINS, EMB, FFT, HOP, LOOK, SR, WINDOW, WNORM


def _compress_ri(re, im, p=0.3):
    """Power compression (re, im) * |z|^(p-1) in FP32 with smooth floor so gradients never spike at silence."""
    s = (re * re + im * im + 1e-10) ** ((p - 1.0) / 2.0)
    return re * s, im * s


class ChannelLayerNorm(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.LayerNorm(ch)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class FreqAttentionBlock(nn.Module):
    """Full-band frequency self-attention + FFN at [B, C, T, F] (zero streaming state, rank <= 5 for Core ML)."""

    def __init__(self, ch, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = ch // heads
        self.norm1 = nn.LayerNorm(ch)
        self.qkv = nn.Linear(ch, 3 * ch)
        self.proj = nn.Linear(ch, ch)
        self.norm2 = nn.LayerNorm(ch)
        self.ffn = nn.Sequential(nn.Linear(ch, ch * 2), nn.SiLU(), nn.Linear(ch * 2, ch))

    def forward(self, x):
        # x: [B, C, T, F] -> xf: [B, T, F, C]
        xf = x.permute(0, 2, 3, 1)
        q, k, v = self.qkv(self.norm1(xf)).chunk(3, dim=-1)
        q = q.unflatten(-1, (self.heads, self.head_dim)).transpose(-3, -2)
        k = k.unflatten(-1, (self.heads, self.head_dim)).transpose(-3, -2)
        v = v.unflatten(-1, (self.heads, self.head_dim)).transpose(-3, -2)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5), dim=-1)
        xf = xf + self.proj((attn @ v).transpose(-3, -2).flatten(-2))
        xf = xf + self.ffn(self.norm2(xf))
        return xf.permute(0, 3, 1, 2)


class RestoreHead(nn.Module):
    """Full-band 481-bin complex restoration & bandwidth-extension head conditioned on noisy X, DFN3 Y, and emb."""

    def __init__(self, ch=(24, 48, 64), n_attn=2, heads=4, emb_dim=EMB):
        super().__init__()
        self.ch = tuple(ch)
        self.n_freq = (BINS - 1) // 4 + 1  # 481 -> 241 -> 121 -> wait: stride 2, 2, 2 gives 61!
        # Encoder: 481 -> 241 -> 121 -> 61 (3 levels, frequency-only)
        self.enc0 = nn.Sequential(nn.Conv2d(4, ch[0], (1, 5), stride=(1, 1), padding=(0, 2)), ChannelLayerNorm(ch[0]), nn.SiLU())
        self.enc1 = nn.Sequential(nn.Conv2d(ch[0], ch[1], (1, 5), stride=(1, 2), padding=(0, 2)), ChannelLayerNorm(ch[1]), nn.SiLU())
        self.enc2 = nn.Sequential(nn.Conv2d(ch[1], ch[2], (1, 5), stride=(1, 2), padding=(0, 2)), ChannelLayerNorm(ch[2]), nn.SiLU())
        self.enc3 = nn.Sequential(nn.Conv2d(ch[2], ch[2], (1, 5), stride=(1, 2), padding=(0, 2)), ChannelLayerNorm(ch[2]), nn.SiLU())
        self.n_mid = 61  # (481-1)//8 + 1

        self.freq_pos = nn.Parameter(torch.zeros(1, ch[2], 1, self.n_mid))
        nn.init.normal_(self.freq_pos, std=0.02)

        # Condition bottleneck on DFN3's temporal encoder embedding (carries multi-second context!)
        self.emb_film = nn.Linear(emb_dim, ch[2] * 2)
        nn.init.zeros_(self.emb_film.weight)
        nn.init.zeros_(self.emb_film.bias)

        # Full-band frequency self-attention blocks + causal per-band temporal GRU
        self.attn_blocks = nn.ModuleList(FreqAttentionBlock(ch[2], heads=heads) for _ in range(n_attn))
        self.t_norm = nn.LayerNorm(ch[2])
        self.t_gru = nn.GRU(ch[2], ch[2], batch_first=True)
        self.t_proj = nn.Linear(ch[2], ch[2])

        # Decoder: 61 -> 121 -> 241 -> 481
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(ch[2], ch[2], (1, 5), stride=(1, 2), padding=(0, 2)), ChannelLayerNorm(ch[2]), nn.SiLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(ch[2], ch[1], (1, 5), stride=(1, 2), padding=(0, 2)), ChannelLayerNorm(ch[1]), nn.SiLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(ch[1], ch[0], (1, 5), stride=(1, 2), padding=(0, 2)), ChannelLayerNorm(ch[0]), nn.SiLU())

        # Output: complex mask delta (dm_re, dm_im) on Y_dfn + additive complex synthesis residual (r_re, r_im)
        # ALL initialized to exact zero -> identity passthrough of DFN3 at step 0!
        self.out = nn.Sequential(
            nn.Conv2d(ch[0], ch[0], (1, 5), padding=(0, 2)),
            ChannelLayerNorm(ch[0]),
            nn.SiLU(),
            nn.Conv2d(ch[0], 4, (1, 3), padding=(0, 1)),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, y_re, y_im, x_re, x_im, emb, hr=None):
        """y_re/im, x_re/im: [B, T, BINS] in libDF WNORM scale; emb: [B, T, EMB]; hr: [1, B*61, ch[-1]]."""
        # Compress features at rfft scale (/ WNORM) so 8-24 kHz harmonics are clearly visible without underflow
        inv_w = 1.0 / WNORM
        with torch.autocast(y_re.device.type, enabled=False):
            yf_re, yf_im = y_re.float(), y_im.float()
            xf_re, xf_im = x_re.float(), x_im.float()
            yc_re, yc_im = _compress_ri(yf_re * inv_w, yf_im * inv_w, 0.3)
            xc_re, xc_im = _compress_ri(xf_re * inv_w, xf_im * inv_w, 0.3)
        inp = torch.stack([yc_re, yc_im, xc_re, xc_im], dim=1)  # [B, 4, T, 481]

        e0 = self.enc0(inp)   # [B, c0, T, 481]
        e1 = self.enc1(e0)    # [B, c1, T, 241]
        e2 = self.enc2(e1)    # [B, c2, T, 121]
        e3 = self.enc3(e2)    # [B, c2, T, 61]

        gamma, beta = self.emb_film(emb).transpose(1, 2).unsqueeze(-1).chunk(2, 1)
        h = e3 * (1.0 + gamma) + beta + self.freq_pos

        for blk in self.attn_blocks:
            h = blk(h)

        # Causal per-band temporal GRU across 61 bands (refines reverb decay & harmonic continuity)
        ht = h.permute(0, 3, 2, 1).flatten(0, 1)  # [B*61, T, c2]
        zt, hr_out = self.t_gru(self.t_norm(ht), hr)
        ht = ht + self.t_proj(zt)
        h = ht.unflatten(0, (-1, self.n_mid)).permute(0, 3, 2, 1)  # [B, c2, T, 61]

        d = self.dec3(h) + e2
        d = self.dec2(d) + e1
        d = self.dec1(d) + e0
        o = self.out(d)  # [B, 4, T, 481]

        with torch.autocast(y_re.device.type, enabled=False):
            o = o.float()
            # Residual complex refinement directly on linear Y_dfn (bit-identical when o == 0!)
            # + scale-equivariant additive synthesis residual (exact 0 on silence, restores missing 8-24 kHz bins!)
            dm_re = 2.0 * torch.tanh(o[:, 0] * 0.5)
            dm_im = 2.0 * torch.tanh(o[:, 1] * 0.5)
            y_energy = (yc_re.detach() * yc_re.detach() + yc_im.detach() * yc_im.detach()).mean(-1, keepdim=True).sqrt()
            rc_re = y_energy * 2.5 * torch.tanh(o[:, 2] * 0.4)
            rc_im = y_energy * 2.5 * torch.tanh(o[:, 3] * 0.4)
            rl_re, rl_im = _compress_ri(rc_re, rc_im, 1.0 / 0.3)

            m_re = 1.0 + dm_re
            m_im = dm_im
            out_re = m_re * yf_re - m_im * yf_im + rl_re * WNORM
            out_im = m_re * yf_im + m_im * yf_re + rl_im * WNORM
        return out_re, out_im, hr_out


class DFN3Studio(nn.Module):
    """Unfrozen DeepFilterNet3 + Full-Band Frequency-Attention RestoreHead."""

    state_names = dfn3.DFN3.state_names + ["h_res"]
    window = WINDOW

    def __init__(self, res_ch=(24, 48, 64), n_attn=2, spk=0):
        super().__init__()
        self.base = dfn3.DFN3(spk=spk)
        self.res_ch = tuple(res_ch)
        self.n_attn = n_attn
        self.restore = RestoreHead(ch=res_ch, n_attn=n_attn)

    def initial_state(self, b=1):
        st = self.base.initial_state(b)
        st["h_res"] = torch.zeros(1, b * self.restore.n_mid, self.res_ch[-1])
        return st

    def step(self, x, **state):
        """One frame [B, 1, 2*BINS] in numpy-scaled rfft (re, im) -> clean frame [B, 1, 2*BINS], then states."""
        if len(state) < len(self.state_names):
            state = self.initial_state(x.shape[0]) | state
        b = self.base
        X = x[:, 0] * WNORM
        re, im = X[:, :BINS], X[:, BINS:]

        # DFN3 features
        e = 10 * torch.log10((re ** 2 + im ** 2) @ b.erb_fb + 1e-10)
        s_erb = e * (1 - dfn3.ALPHA) + (state["s_erb"] + dfn3.ERB_INIT) * dfn3.ALPHA
        mag = (re[:, :dfn3.NB_DF] ** 2 + im[:, :dfn3.NB_DF] ** 2).sqrt()
        s_spec = mag * (1 - dfn3.ALPHA) + (state["s_spec"] + dfn3.SPEC_INIT) * dfn3.ALPHA
        norm = s_spec.clamp(min=1e-12).sqrt()
        f_erb = torch.cat([state["f_erb"], ((e - s_erb) / 40)[:, None, None]], 2)
        f_spec = torch.cat([state["f_spec"], torch.stack([re[:, :dfn3.NB_DF] / norm, im[:, :dfn3.NB_DF] / norm], 1)[:, :, None]], 2)

        # DFN3 encoder
        enc = b.enc
        e0 = enc.erb_conv0.frame(f_erb)
        e1 = enc.erb_conv1.frame(e0)
        e2 = enc.erb_conv2.frame(e1)
        e3 = enc.erb_conv3.frame(e2)
        c0 = enc.df_conv0.frame(f_spec)
        c1 = enc.df_conv1.frame(c0)
        emb = e3.permute(0, 2, 3, 1).flatten(2) + enc.df_fc_emb(c1.permute(0, 2, 3, 1).flatten(2))
        emb, h_enc = enc.emb_gru(emb, state["h_enc"])

        # DFN3 ERB decoder + boost head
        d = b.erb_dec
        y, h_erb = d.emb_gru(emb, state["h_erb"])
        y = y.unflatten(-1, (dfn3.NB_ERB // 4, dfn3.CH)).permute(0, 3, 1, 2)
        y = d.convt3.frame(d.conv3p.frame(e3) + y)
        y = d.convt2(d.conv2p.frame(e2) + y)
        y = d.convt1(d.conv1p.frame(e1) + y)
        m = d.gains(d.conv0p.frame(e0) + y)[:, 0]

        # DFN3 deep filter decoder
        dd = b.df_dec
        c, h_df = dd.df_gru(emb, state["h_df"])
        c = c + dd.df_skip(emb)
        c0s = torch.cat([state["c0"], c0], 2)
        coefs = dd.df_out(c)[:, 0].unflatten(-1, (dfn3.NB_DF, dfn3.DF_ORDER * 2)) + dd.df_convp.frame(c0s)[:, :, 0].transpose(1, 2)
        c_re, c_im = coefs[..., 0::2], coefs[..., 1::2]

        # DFN3 first-stage complex output at t - LOOK
        specs = torch.cat([state["spec"], X[:, None]], 1)
        o_re, o_im = specs[:, LOOK, :BINS], specs[:, LOOK, BINS:]
        g = m @ b.mask.erb_inv_fb
        h_re, h_im = specs[..., :dfn3.NB_DF].transpose(1, 2), specs[..., BINS:BINS + dfn3.NB_DF].transpose(1, 2)
        df_re = (h_re * c_re - h_im * c_im).sum(-1)
        df_im = (h_re * c_im + h_im * c_re).sum(-1)
        y1_re = torch.cat([df_re, (g * o_re)[:, dfn3.NB_DF:]], 1)
        y1_im = torch.cat([df_im, (g * o_im)[:, dfn3.NB_DF:]], 1)

        # Second-stage full-band studio restoration + BWE head (zero-init -> exact DFN3 at step 0!)
        out_re, out_im, h_res = self.restore(y1_re[:, None], y1_im[:, None], o_re[:, None], o_im[:, None], emb, state["h_res"])
        y_out = torch.cat([out_re[:, 0], out_im[:, 0]], 1)[:, None] / WNORM

        return (
            y_out,
            s_erb - dfn3.ERB_INIT,
            s_spec - dfn3.SPEC_INIT,
            f_erb[:, :, 1:],
            f_spec[:, :, 1:],
            c0s[:, :, 1:],
            specs[:, 1:],
            h_enc,
            h_erb,
            h_df,
            h_res,
        )

    def forward(self, wav):
        """Full-clip training forward pass matching step() frame-by-frame."""
        b = self.base
        n = wav.shape[1]
        with torch.autocast(wav.device.type, enabled=False):
            wav_f = wav.float()
            x = F.pad(wav_f, (HOP, HOP * (LOOK + 1) + (-n) % HOP))
            win = WINDOW.to(wav.device)
            X = torch.fft.rfft(x.unfold(1, FFT, HOP) * win) * WNORM
            re, im = X.real, X.imag
            with torch.no_grad():
                e = 10 * torch.log10((re ** 2 + im ** 2) @ b.erb_fb + 1e-10)
                mag = (re[..., :dfn3.NB_DF] ** 2 + im[..., :dfn3.NB_DF] ** 2).sqrt()
                s_erb, s_spec = (dfn3.ema(v, init.to(v.device)) for v, init in ((e, dfn3.ERB_INIT), (mag, dfn3.SPEC_INIT)))
                norm = s_spec.clamp(min=1e-12).sqrt()
                f_erb = ((e - s_erb) / 40)[:, None]
                f_spec = torch.stack([re[..., :dfn3.NB_DF] / norm, im[..., :dfn3.NB_DF] / norm], 1)

        enc = b.enc
        e0 = enc.erb_conv0(f_erb)
        e1 = enc.erb_conv1(e0)
        e2 = enc.erb_conv2(e1)
        e3 = enc.erb_conv3(e2)
        c0 = enc.df_conv0(f_spec)
        c1 = enc.df_conv1(c0)
        emb = e3.permute(0, 2, 3, 1).flatten(2) + enc.df_fc_emb(c1.permute(0, 2, 3, 1).flatten(2))
        emb, _ = enc.emb_gru(emb)

        d = b.erb_dec
        y, _ = d.emb_gru(emb)
        y = y.unflatten(-1, (dfn3.NB_ERB // 4, dfn3.CH)).permute(0, 3, 1, 2)
        y = d.convt3(d.conv3p(e3) + y)
        y = d.convt2(d.conv2p(e2) + y)
        y = d.convt1(d.conv1p(e1) + y)
        m = d.gains(d.conv0p(e0) + y)

        dd = b.df_dec
        c, _ = dd.df_gru(emb)
        c = c + dd.df_skip(emb)
        coefs = dd.df_out(c).unflatten(-1, (dfn3.NB_DF, dfn3.DF_ORDER * 2)) + dd.df_convp(c0).permute(0, 2, 3, 1)
        c_re, c_im = coefs[..., 0::2], coefs[..., 1::2]

        with torch.autocast(wav.device.type, enabled=False):
            T = X.shape[1]
            low = F.pad(torch.stack([re[..., :dfn3.NB_DF], im[..., :dfn3.NB_DF]], -1), (0, 0, 0, 0, dfn3.DF_ORDER - 1, 0))
            h = low.unfold(1, dfn3.DF_ORDER, 1)
            df_re = (h[..., 0, :] * c_re.float() - h[..., 1, :] * c_im.float()).sum(-1)
            df_im = (h[..., 0, :] * c_im.float() + h[..., 1, :] * c_re.float()).sum(-1)
            g = m.float() @ b.mask.erb_inv_fb
            o_re, o_im = (F.pad(a, (0, 0, LOOK, 0))[:, :T] for a in (re, im))
            y1_re = torch.cat([df_re, (g * o_re)[..., dfn3.NB_DF:]], -1)
            y1_im = torch.cat([df_im, (g * o_im)[..., dfn3.NB_DF:]], -1)

        out_re, out_im, _ = self.restore(y1_re, y1_im, o_re, o_im, emb)

        with torch.autocast(wav.device.type, enabled=False):
            Y = torch.complex(out_re.float(), out_im.float())
            frames = torch.fft.irfft(Y[:, LOOK:] / WNORM, n=FFT) * win
            return (frames[:, 1:, :HOP] + frames[:, :-1, HOP:]).flatten(1)[:, :n]


def init_from_base(base_ckpt, res_ch=(24, 48, 64), n_attn=2):
    """Initialize DFN3Studio from CleanMyMic Base (hold-10000.pt); bit-identical at step 0."""
    base_model = dfn3.load(base_ckpt)
    model = DFN3Studio(res_ch=res_ch, n_attn=n_attn, spk=base_model.spk)
    model.base.load_state_dict(base_model.state_dict())
    return model


def load(path, map_location="cpu"):
    state = torch.load(path, map_location=map_location, weights_only=False)
    cfg = state.get("config", {})
    model = DFN3Studio(
        res_ch=tuple(cfg.get("res_ch", (24, 48, 64))),
        n_attn=cfg.get("n_attn", 2),
        spk=cfg.get("spk", 0),
    )
    model.load_state_dict(state["model"])
    return model.eval(), state


def stream(model, wav):
    """Exact frame-by-frame mirror of Enhancer.swift for DFN3Studio."""
    buf, tail, state, out = torch.zeros(FFT), torch.zeros(HOP), {}, []
    with torch.no_grad():
        for i in range(0, len(wav) - HOP + 1, HOP):
            buf = torch.cat([buf[HOP:], wav[i:i + HOP]])
            X = torch.fft.rfft(buf * WINDOW)
            y, *s = model.step(torch.cat([X.real, X.imag])[None, None], **state)
            state = dict(zip(model.state_names, s))
            frame = torch.fft.irfft(torch.complex(*y[0, 0].chunk(2)), n=FFT) * WINDOW
            out.append(tail + frame[:HOP])
            tail = frame[HOP:]
    return torch.cat(out)


if __name__ == "__main__":
    torch.manual_seed(0)
    t = torch.arange(3 * SR) / SR
    wav = (0.1 * torch.sin(2 * math.pi * 220 * t) * (1 + torch.sin(2 * math.pi * 3 * t)) * (t > 1) + 0.01 * torch.randn(3 * SR))
    m = DFN3Studio().eval()
    n_tot = sum(p.numel() for p in m.parameters()) / 1e6
    n_res = sum(p.numel() for p in m.restore.parameters()) / 1e6
    lag = HOP * (1 + LOOK)
    with torch.no_grad():
        # Step 0 must match plain base DFN3
        y0, yb = m(wav[None])[0], m.base(wav[None])[0]
        id_snr = 10 * math.log10((yb.pow(2).sum() / (y0 - yb).pow(2).sum().clamp(min=1e-20)).item())
        assert id_snr > 60, f"step 0 differs from Base: {id_snr:.1f} dB"
        # After perturbing RestoreHead, forward() == stream()
        m.restore.out[-1].weight.normal_(0, 0.03)
        m.restore.out[-1].bias.normal_(0, 0.03)
        a, b = m(wav[None])[0, :-lag], stream(m, wav)[lag:]
        snr = 10 * math.log10((b.pow(2).sum() / (a - b).pow(2).sum().clamp(min=1e-20)).item())
        assert snr > 60, f"forward != stream: {snr:.1f} dB"
        sil = m(torch.zeros(1, SR))[0]
        assert torch.isfinite(sil).all() and sil.abs().max() < 1e-6, f"silence bad: {sil.abs().max()}"
    print(f"ok DFN3Studio: {n_tot:.2f}M total ({n_res:.2f}M RestoreHead), init==Base {id_snr:.1f} dB, stream==fwd {snr:.1f} dB")
