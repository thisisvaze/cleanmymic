"""Unit tests for StudioNet, StudioNetV2, studio_data, prep_teacher, and StudioLoss on CPU."""

import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from model import SR, load as load_ckpt
from prep_teacher import gate_pair
from studio_data import TeacherPairPool, align_rirs, make_studio_batch
from studio_net import StudioNet, StudioNetV2, stream
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


# 1. RIR peak alignment (RE-USE shifted anechoic target invariant)
unaligned = torch.zeros(3, 1000)
unaligned[0, 120] = 1.0
unaligned[1, 35] = -0.9
unaligned[2, 0] = 0.8
aligned = align_rirs(unaligned)
assert (aligned.abs().argmax(dim=1) == 0).all(), (
    "align_rirs failed to align direct-path peaks"
)

# 2. Dry anechoic full-band targets + TeacherPairPool integration
torch.manual_seed(0)
clean, noise = _fake_pool(), _fake_pool()
rirs = torch.zeros(4, 4800)
rirs[:, 0] = 1.0
rirs[:, 1200] = 0.5
mic_irs = torch.zeros(3, 512)
mic_irs[:, 0] = 1.0
mic_irs[:, 2] = -0.3

with tempfile.TemporaryDirectory() as td:
    tp_root = Path(td)
    (tp_root / "noisy").mkdir()
    (tp_root / "clean").mkdir()
    t_ax = np.arange(2 * SR, dtype=np.float32) / SR
    sig = 0.2 * np.sin(2 * np.pi * 300.0 * t_ax) + 0.05 * np.sin(
        2 * np.pi * 12000.0 * t_ax
    )
    sf.write(
        tp_root / "noisy" / "clip0.flac",
        sig + 0.03 * np.random.randn(2 * SR).astype(np.float32),
        SR,
    )
    sf.write(tp_root / "clean" / "clip0.flac", sig, SR)
    tpool = TeacherPairPool(tp_root, device="cpu")
    assert len(tpool) == 1
    xtp, ttp = make_studio_batch(
        clean, noise, rirs, 4, SR, mic_irs=mic_irs, teacher_pairs=tpool, clean_prob=0.0
    )
    assert (
        xtp.shape == (4, SR) and torch.isfinite(xtp).all() and torch.isfinite(ttp).all()
    )

    # gate_pair sanity checks: good full-band pair passes, collapsed/band-limited fails
    ok_good, reasons_good, _ = gate_pair(sig, sig, asr_wer=0.02, spk_sim=0.96)
    assert ok_good and not reasons_good, f"good pair rejected: {reasons_good}"
    ok_bad, reasons_bad, _ = gate_pair(sig, sig * 1e-4, asr_wer=0.40, spk_sim=0.50)
    assert not ok_bad and len(reasons_bad) >= 3

x, t = make_studio_batch(
    clean, noise, rirs, 4, SR, mic_irs=mic_irs, clean_prob=0.0, force_bandlim_hz=4000.0
)
assert x.shape == (4, SR) and t.shape == (4, SR)
assert torch.isfinite(x).all() and torch.isfinite(t).all()
assert t.abs().max() <= 0.921
f = torch.fft.rfftfreq(SR, 1 / SR)
hi = (f >= 8000) & (f <= 20000)
e_x = torch.fft.rfft(x)[:, hi].abs().pow(2).mean()
e_t = torch.fft.rfft(t)[:, hi].abs().pow(2).mean()
assert e_t > 20.0 * e_x, f"target lost high-band energy: e_t={e_t}, e_x={e_x}"

# Clean identity pairs: clean_prob=1.0 -> exact x == t
xc, tc = make_studio_batch(clean, noise, rirs, 4, SR, clean_prob=1.0)
assert torch.allclose(xc, tc), "clean identity pairs not identical"

# Pure non-speech / distant-chatter rows: noise_only_prob=1.0 -> target == 0, non-zero ns floor loss
xns, tns = make_studio_batch(
    clean, noise, rirs, 4, SR, events=noise, clean_prob=0.0, noise_only_prob=1.0
)
assert tns.abs().max() == 0.0 and xns.abs().max() > 1e-4

# 3. StudioNet (with optional speaker FiLM) + StudioLoss + discriminators
m = StudioNet(ch=(16, 24, 32, 32), n_blocks=2, heads=4, global_hidden=64, spk_dim=32)
crit = StudioLoss(asym=0.35, ns_weight=2.0, wavlm_weight=0.0, device="cpu")
disc = HighBandDisc(1024, ch=16)
pdisc = PeriodDisc(3, ch=16)
spk = F.normalize(torch.randn(4, 32), dim=-1)

y = m(x, spk=spk)
loss, parts = crit(y, t, x=x, use_wavlm=False)
loss_ns, parts_ns = crit(m(xns), tns, x=xns, use_wavlm=False)
assert parts_ns["ns"] > 0.0, (
    "non-speech residual penalty should activate on unattenuated noise"
)
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
    offline, streamed = m(x[:1], spk=spk[:1])[0], stream(m, x[0], spk=spk[:1])
lag = 480 * (1 + m.look)
err = (streamed[lag:] - offline[: len(streamed) - lag]).abs().max().item()
assert err < 1e-4, f"stream != offline: {err}"

# 4. Two-stage StudioNetV2: init == Stage 1, freeze_stage1() protects Stage 1, streaming parity, load roundtrip
s1_cfg = dict(
    ch=(16, 24, 32, 32),
    n_blocks=2,
    heads=4,
    global_hidden=64,
    look=2,
    past=2,
    df=True,
    spk_dim=32,
)
v2 = StudioNetV2(s1_cfg=s1_cfg, ref_ch=(16, 24, 32), ref_blocks=2, ref_heads=4)
v2.stage1.load_state_dict(m.state_dict())
v2.eval()
with torch.no_grad():
    y_s1 = v2.stage1(x[:1], spk=spk[:1])
    y_v2_0 = v2(x[:1], spk=spk[:1])
    assert (y_v2_0 - y_s1).abs().max().item() < 1e-5, (
        "StudioNetV2 step-0 output must match Stage 1"
    )

v2.train()
v2.freeze_stage1()
y_v2 = v2(x, spk=spk)
loss_v2, _ = crit(y_v2, t, x=x, use_wavlm=False)
loss_v2.backward()
assert all(p.grad is None for p in v2.stage1.parameters()), (
    "frozen Stage 1 received gradients!"
)
assert any(
    p.grad is not None and p.grad.abs().sum() > 0 for p in v2.refiner.parameters()
), "refiner got no grad!"

v2.eval()
with torch.no_grad():
    v2.refiner.out[-1].weight.normal_(0, 0.02)
    v2.refiner.out[-1].bias.normal_(0, 0.02)
    assert v2(torch.zeros(1, SR), spk=spk[:1]).abs().max() == 0.0
    v2_off, v2_str = v2(x[:1], spk=spk[:1])[0], stream(v2, x[0], spk=spk[:1])
v2_err = (v2_str[lag:] - v2_off[: len(v2_str) - lag]).abs().max().item()
assert v2_err < 1e-4, f"StudioNetV2 stream != offline: {v2_err}"

with tempfile.TemporaryDirectory() as td:
    ckpt_path = Path(td) / "v2.pt"
    cfg = {
        "arch": "studio_v2",
        "s1_cfg": s1_cfg,
        "ref_ch": (16, 24, 32),
        "ref_blocks": 2,
        "ref_heads": 4,
    }
    torch.save({"model": v2.state_dict(), "config": cfg}, ckpt_path)
    loaded, _ = load_ckpt(ckpt_path)
    with torch.no_grad():
        assert torch.allclose(loaded(x[:1], spk=spk[:1]), v2_off[None], atol=1e-5)

print(
    f"ok test_studio: high-band={e_t / e_x:.0f}x, loss={loss.item():.3f}, "
    f"s1_err={err:.1e}, v2_err={v2_err:.1e}, frozen_s1=ok"
)
