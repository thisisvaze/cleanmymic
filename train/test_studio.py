"""Unit tests for StudioNet, studio_data, and StudioLoss on CPU with synthetic data."""

import math
from types import SimpleNamespace

import torch

from model import SR
from studio_data import make_studio_batch
from studio_net import StudioNet, stream
from train_studio import HighBandDisc, PeriodDisc, StudioLoss


def _fake_pool(n_samples=SR):
    def sample(n, seg, wrap=False, flags=False):
        t = torch.arange(seg).float() / SR
        x = 0.2 * torch.sin(2 * math.pi * 440.0 * t)[None].repeat(
            n, 1
        ) + 0.05 * torch.randn(n, seg)
        return (x, torch.zeros(n, dtype=torch.bool)) if flags else x

    def joined(n, seg, **kwargs):
        t = torch.arange(seg).float() / SR
        return 0.2 * torch.sin(2 * math.pi * 220.0 * t)[None].repeat(
            n, 1
        ) + 0.02 * torch.randn(n, seg)

    return SimpleNamespace(sample=sample, joined=joined)


# 1. Dry anechoic full-band targets: input is bandlimited & reverberant, target stays dry & full-band
torch.manual_seed(0)
clean, noise = _fake_pool(), _fake_pool()
rirs = torch.zeros(4, 4800)
rirs[:, 0] = 1.0
rirs[:, 1200] = 0.5
mic_irs = torch.zeros(3, 512)
mic_irs[:, 0] = 1.0
mic_irs[:, 2] = -0.3

x, t = make_studio_batch(
    clean, noise, rirs, 4, SR, mic_irs=mic_irs, clean_prob=0.0, force_bandlim_hz=4000.0
)
assert x.shape == (4, SR) and t.shape == (4, SR)
assert torch.isfinite(x).all() and torch.isfinite(t).all()
assert t.abs().max() <= 0.921
# High-band (8-20 kHz) energy must be stripped in x (forced 4 kHz lowpass) while preserved in full-band target t
f = torch.fft.rfftfreq(SR, 1 / SR)
hi = (f >= 8000) & (f <= 20000)
e_x = torch.fft.rfft(x)[:, hi].abs().pow(2).mean()
e_t = torch.fft.rfft(t)[:, hi].abs().pow(2).mean()
assert e_t > 20.0 * e_x, f"target lost high-band energy: e_t={e_t}, e_x={e_x}"

# Clean identity pairs: clean_prob=1.0 -> exact x == t
xc, tc = make_studio_batch(clean, noise, rirs, 4, SR, clean_prob=1.0)
assert torch.allclose(xc, tc), "clean identity pairs not identical"

# Pure non-speech noise-only rows: noise_only_prob=1.0 -> target == 0, finite loss
xns, tns = make_studio_batch(
    clean, noise, rirs, 4, SR, clean_prob=0.0, noise_only_prob=1.0
)
assert tns.abs().max() == 0.0 and xns.abs().max() > 1e-4

# 2. StudioNet forward/stream parity, silence=0, backward pass, StudioLoss, HighBandDisc & PeriodDisc
m = StudioNet(ch=(16, 24, 32, 32), n_blocks=2, heads=4, global_hidden=64)
crit = StudioLoss(asym=0.35, wavlm_weight=0.0, device="cpu")
disc = HighBandDisc(1024, ch=16)
pdisc = PeriodDisc(3, ch=16)

y = m(x)
loss, parts = crit(y, t, x=x, use_wavlm=False)
loss_ns, _ = crit(m(xns), tns, x=xns, use_wavlm=False)
logit, feats = disc(y)
plogit, pfeats = pdisc(y)
assert (
    torch.isfinite(loss)
    and torch.isfinite(loss_ns)
    and logit.ndim == 4
    and len(feats) == 5
    and len(pfeats) == 5
)
(loss + loss_ns * 0.1 + logit.mean() * 0.01 + plogit.mean() * 0.01).backward()
assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())

m.eval()
with torch.no_grad():
    assert m(torch.zeros(1, SR)).abs().max() == 0.0
    offline, streamed = m(x[:1])[0], stream(m, x[0])
lag = 480 * (1 + m.look)
err = (streamed[lag:] - offline[: len(streamed) - lag]).abs().max().item()
assert err < 1e-4, f"stream != offline: {err}"
print(
    f"ok test_studio: high-band target/in ratio={e_t / e_x:.0f}x, loss={loss.item():.3f}, stream err={err:.1e}"
)
