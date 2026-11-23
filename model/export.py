"""Export a PyTorch speech enhancement checkpoint to a streaming Core ML .mlpackage.

One 10 ms STFT frame in, one out, with explicit recurrent state passed in and out.
STFT and overlap-add iSTFT stay outside the model (handled in Swift with vDSP).

Usage:
    uv run model/export.py <checkpoint.pt | model_120.ckpt.best> <out.mlpackage> [--precision fp32|mixed]

Verifies silence/sparse finiteness, 200-frame PyTorch-vs-Core ML parity, and times per-frame latency.
"""

import argparse
import time

import coremltools as ct
import numpy as np
import torch

import dfn3
import dfn3_studio
from dfn3 import BINS


class Step(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, *state):
        return self.model.step(x, **dict(zip(self.model.state_names, state)))


def _fp16_heavy_ops(op):
    return op.op_type in ("linear", "matmul", "gru")


def export_model(model, out, meta, precision="fp32", bench=True):
    x = torch.zeros(1, 1, 2 * BINS)
    with torch.no_grad():
        _, *state = model.step(x)
    names = ["x"] + list(model.state_names)
    example = (x,) + tuple(torch.zeros_like(a) for a in state)
    traced = torch.jit.trace(Step(model).eval(), example)
    compute_prec = (
        ct.transform.FP16ComputePrecision(op_selector=_fp16_heavy_ops)
        if precision == "mixed"
        else ct.precision.FLOAT32
    )
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(n, shape=a.shape) for n, a in zip(names, example)],
        outputs=[
            ct.TensorType(n if n == "y" else f"{n}_out") for n in ["y"] + names[1:]
        ],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=compute_prec,
    )
    mlmodel.user_defined_metadata.update(meta)
    mlmodel.save(out)

    zero_st = {n: a.numpy() for n, a in zip(names[1:], example[1:])}
    sparse = np.zeros((1, 1, 2 * BINS), np.float32)
    sparse[0, 0, 10] = 0.2
    for label, probe in (
        ("silence", np.zeros((1, 1, 2 * BINS), np.float32)),
        ("sparse", sparse),
    ):
        res = mlmodel.predict({"x": probe} | zero_st)
        bad = [k for k, v in res.items() if not np.isfinite(v).all()]
        assert not bad, f"Core ML ({precision}) non-finite on {label}: {bad}"

    torch.manual_seed(0)
    frames = torch.randn(200, 1, 1, 2 * BINS) * 0.5
    ts = example[1:]
    state_np = {n: a.numpy() for n, a in zip(names[1:], example[1:])}
    worst = 0.0
    with torch.no_grad():
        for f in frames:
            ty, *ts = model.step(f, **dict(zip(names[1:], ts)))
            res = mlmodel.predict({"x": f.numpy()} | state_np)
            state_np = {n: res[f"{n}_out"] for n in names[1:]}
            worst = max(
                worst,
                float(
                    np.abs(res["y"] - ty.numpy()).max() / (ty.abs().max().item() + 1e-6)
                ),
            )
    assert worst < 0.05, (
        f"Core ML output differs from PyTorch (worst relative error {worst:.3f})"
    )
    print(
        f"ok: Core ML ({precision}) matches PyTorch over 200 frames (worst rel err {worst:.4f})"
    )

    if bench:
        for unit in ("CPU_ONLY", "CPU_AND_NE", "ALL"):
            m = ct.models.MLModel(out, compute_units=getattr(ct.ComputeUnit, unit))
            inp = {"x": frames[0].numpy()} | {
                n: a.numpy() for n, a in zip(names[1:], example[1:])
            }
            for _ in range(20):
                m.predict(inp)
            t = time.perf_counter()
            for _ in range(500):
                m.predict(inp)
            ms = (time.perf_counter() - t) / 500 * 1000
            print(f"{unit:>10}: {ms:.3f} ms per 10 ms frame (RTF {ms / 10:.3f})")
    return worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="checkpoint (.pt or .ckpt.best)")
    ap.add_argument("out", help="output .mlpackage path")
    ap.add_argument("--precision", choices=("fp32", "mixed"), default="fp32")
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()

    if ".ckpt" in args.ckpt:
        model = dfn3.load(args.ckpt)
        engine = "DeepFilterNet3" if args.ckpt.endswith(".best") else "CleanMyMic"
    else:
        model, _ = dfn3_studio.load(args.ckpt)
        engine = "CleanMyMic"

    meta = {
        "engine": engine,
        "window": "vorbis",
        "compress": "1",
        "look": str(dfn3.LOOK),
    }
    export_model(
        model.eval(), args.out, meta, precision=args.precision, bench=not args.no_bench
    )
