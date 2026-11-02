"""Export a checkpoint to Core ML: one 10 ms frame in, one out. State is passed in and out explicitly: the GRU's (h),
the last HIST input frames (buf), which the deep filter and the lookahead read, and for floor models the floor
history (fl). The Mac app starts every state at zeros.
STFT/iSTFT stay outside the model (the Mac app does them with vDSP, like model.stream()).

    uv run train/export.py <checkpoint.pt | none> <out.mlpackage>
    uv run train/export.py models/dfn3/model_120.ckpt.best models/dfn3.mlpackage   # the DeepFilterNet3 engine
    uv run train/export.py models/dfn-far-60.ckpt models/talkers-e60.mlpackage     # a dfn-recipe run of ours

Checks the Core ML output against PyTorch frame by frame and times one prediction per compute unit. The package's
metadata tells the app how to frame the audio for it: window ("sqrt-hann" or "vorbis"), compress (exponent) and look
(frames of lookahead, to line up the dry signal for the strength mix).
"""
import sys
import time

import coremltools as ct
import numpy as np
import torch

import dfn3
from model import BINS, Model, load


class Step(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, *state):
        return self.model.step(x, **dict(zip(self.model.state_names, state)))


if __name__ == "__main__":
    ckpt, out = sys.argv[1], sys.argv[2]
    if ".ckpt" in ckpt:  # DeepFilterNet's format: its released DFN3 (.ckpt.best), or a dfn-recipe run of ours
        model = dfn3.load(ckpt)
        meta = {"engine": "DeepFilterNet3" if ckpt.endswith(".best") else "CleanMyMic", "window": "vorbis",
                "compress": "1", "look": str(dfn3.LOOK)}
    else:
        model = (load(ckpt)[0] if ckpt != "none" else Model()).eval()
        meta = ({"engine": "CleanMyMic", "window": "vorbis", "compress": "1", "look": str(dfn3.LOOK)}
                if isinstance(model, dfn3.DFN3) else  # the studio stage: DeepFilterNet3's network and framing
                {"engine": "CleanMyMic", "window": "sqrt-hann", "compress": "0.3", "look": str(model.look)})
    if ckpt == "none":
        with torch.no_grad():
            model.out.weight.normal_(0, 0.01)  # untrained but not passthrough, so the check means something
    x = torch.zeros(1, 1, 2 * BINS)
    with torch.no_grad():
        _, *state = model.step(x)  # the state's shapes; it starts at zeros, as in the Mac app
    names, example = ["x"] + model.state_names, (x,) + tuple(torch.zeros_like(a) for a in state)
    traced = torch.jit.trace(Step(model).eval(), example)
    mlmodel = ct.convert(traced, inputs=[ct.TensorType(n, shape=a.shape) for n, a in zip(names, example)],
                         outputs=[ct.TensorType(n if n == "y" else f"{n}_out") for n in ["y"] + names[1:]],
                         minimum_deployment_target=ct.target.macOS15,
                         # FP16 underflows |Y|^2 in quiet bins and _power's ^-0.35 blows up (trained v4: error 1.7)
                         compute_precision=ct.precision.FLOAT32)
    mlmodel.user_defined_metadata.update(meta)
    mlmodel.save(out)

    torch.manual_seed(0)
    frames = torch.randn(200, 1, 1, 2 * BINS) * 0.5
    ts, state, worst = example[1:], {n: a.numpy() for n, a in zip(names[1:], example[1:])}, 0.0
    with torch.no_grad():
        for f in frames:
            ty, *ts = model.step(f, **dict(zip(names[1:], ts)))
            res = mlmodel.predict({"x": f.numpy()} | state)
            state = {n: res[f"{n}_out"] for n in names[1:]}
            worst = max(worst, float(np.abs(res["y"] - ty.numpy()).max() / (ty.abs().max().item() + 1e-6)))
    assert worst < 0.05, f"Core ML output differs from PyTorch (worst relative error {worst:.3f})"
    print(f"ok: Core ML matches PyTorch over 200 frames (worst relative error {worst:.4f})")

    for unit in ("CPU_ONLY", "CPU_AND_NE", "ALL"):
        m = ct.models.MLModel(out, compute_units=getattr(ct.ComputeUnit, unit))
        inp = {"x": frames[0].numpy()} | {n: a.numpy() for n, a in zip(names[1:], example[1:])}
        for _ in range(20):
            m.predict(inp)
        t = time.perf_counter()
        for _ in range(500):
            m.predict(inp)
        ms = (time.perf_counter() - t) / 500 * 1000
        print(f"{unit:>10}: {ms:.3f} ms per 10 ms frame (real-time factor {ms / 10:.3f})")
