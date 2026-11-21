"""Export a checkpoint to Core ML: one 10 ms frame in, one out. State is passed in and out explicitly: the GRU's (h),
the last HIST input frames (buf), which the deep filter and the lookahead read, and for floor models the floor
history (fl). The Mac app starts every state at zeros.
STFT/iSTFT stay outside the model (the Mac app does them with vDSP, like model.stream()).

    uv run train/export.py <checkpoint.pt | none> <out.mlpackage> [--precision fp32|mixed]
    uv run train/export.py models/dfn3/model_120.ckpt.best models/dfn3.mlpackage   # the DeepFilterNet3 engine
    uv run train/export.py models/dfn-far-60.ckpt models/talkers-e60.mlpackage     # a dfn-recipe run of ours

Checks the Core ML output against PyTorch frame by frame and times one prediction per compute unit. The package's
metadata tells the app how to frame the audio for it: window ("sqrt-hann" or "vorbis"), compress (exponent) and look
(frames of lookahead, to line up the dry signal for the strength mix).
"""

import argparse
import time

import coremltools as ct
import numpy as np
import torch

import dfn3
import dfn3_studio
from model import BINS, Model, load


class Step(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, *state):
        return self.model.step(x, **dict(zip(self.model.state_names, state)))


def _fp16_heavy_ops(op):
    """Selective FP16: cast attention/FFN/GRU ops (>90% of params) to FP16 while keeping convs & spectral math FP32."""
    return op.op_type in ("linear", "matmul", "gru")


def export_model(model, out, meta, precision="fp32", bench=True):
    x = torch.zeros(1, 1, 2 * BINS)
    with torch.no_grad():
        _, *state = model.step(
            x
        )  # the state's shapes; it starts at zeros, as in the Mac app
    names, example = (
        ["x"] + model.state_names,
        (x,) + tuple(torch.zeros_like(a) for a in state),
    )
    traced = torch.jit.trace(Step(model).eval(), example)
    # Full FP16 underflows |Y|^2 in quiet bins and _power's ^-0.35 blows up; mixed keeps linear/matmul/GRU FP16.
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

    # Verify exact digital silence and sparse single-bin tone stay 100% finite (catches FP16 0*inf NaNs)
    zero_st = {n: a.numpy() for n, a in zip(names[1:], example[1:])}
    sparse = np.zeros((1, 1, 2 * BINS), np.float32)
    sparse[0, 0, 10] = 0.2
    for label, probe in (
        ("silence", np.zeros((1, 1, 2 * BINS), np.float32)),
        ("sparse", sparse),
    ):
        res = mlmodel.predict({"x": probe} | zero_st)
        bad = [k for k, v in res.items() if not np.isfinite(v).all()]
        assert not bad, (
            f"Core ML ({precision}) produced non-finite values on {label}: {bad}"
        )

    torch.manual_seed(0)
    frames = torch.randn(200, 1, 1, 2 * BINS) * 0.5
    ts, state_np, worst = (
        example[1:],
        {n: a.numpy() for n, a in zip(names[1:], example[1:])},
        0.0,
    )
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
        f"ok: Core ML ({precision}) matches PyTorch over 200 frames + silence/sparse finite (worst rel err {worst:.4f})"
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
            print(
                f"{unit:>10}: {ms:.3f} ms per 10 ms frame (real-time factor {ms / 10:.3f})"
            )
    return worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="checkpoint (.pt or .ckpt) or 'none'")
    ap.add_argument("out", help="output .mlpackage path")
    ap.add_argument("--precision", choices=("fp32", "mixed"), default="fp32")
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()

    if (
        ".ckpt" in args.ckpt
    ):  # DeepFilterNet's format: its released DFN3 (.ckpt.best), or a dfn-recipe run of ours
        model = dfn3.load(args.ckpt)
        meta = {
            "engine": "DeepFilterNet3" if args.ckpt.endswith(".best") else "CleanMyMic",
            "window": "vorbis",
            "compress": "1",
            "look": str(dfn3.LOOK),
        }
    else:
        model = (load(args.ckpt)[0] if args.ckpt != "none" else Model()).eval()
        meta = (
            {
                "engine": "CleanMyMic",
                "window": "vorbis",
                "compress": "1",
                "look": str(dfn3.LOOK),
            }
            if isinstance(model, (dfn3.DFN3, dfn3_studio.DFN3Studio))
            else {
                "engine": "CleanMyMic",
                "window": "sqrt-hann",
                "compress": "0.3",
                "look": str(model.look),
            }
        )
    if args.ckpt == "none":
        with torch.no_grad():
            model.out.weight.normal_(
                0, 0.01
            )  # untrained but not passthrough, so the check means something
    export_model(
        model, args.out, meta, precision=args.precision, bench=not args.no_bench
    )
