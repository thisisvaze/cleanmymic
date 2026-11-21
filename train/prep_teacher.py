"""Build fidelity-gated offline teacher pairs for StudioNet v2 distillation.

Designed around commercial-ready 48 kHz studio restorers such as NVIDIA's
`nvidia/NVIDIA-NemotronLabs-RE-USE` (OpenMDW-1.1: commercial use permitted, no
restrictions on outputs). Because `mamba-ssm` typically runs inside NVIDIA's
CUDA container, this script supports two workflows:

1) Gate pre-generated teacher outputs (recommended):
     uv run --group eval train/prep_teacher.py \
       --noisy-dir ~/data/real_noisy \
       --teacher-dir ~/data/reuse_out \
       --out ~/experiments/studionet-v2/teacher_pairs

2) Optionally verify outputs using only acoustic/duration/high-band checks (no ASR/WavLM):
     uv run train/prep_teacher.py --noisy-dir ... --teacher-dir ... --out ... --no-neural-gates

Every clip is checked against SPEC.md's fidelity gates before being copied into
`<out>/noisy/` and `<out>/clean/` (consumed by `TeacherPairPool` in `train/studio_data.py`),
and every decision (accepted or refused + scores) is logged to `<out>/manifest.jsonl`.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 48_000


def _read_mono_48k(path: Path) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        g = int(np.gcd(sr, SR))
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def _rms_db(x: np.ndarray) -> float:
    return float(10.0 * np.log10(np.mean(x * x) + 1e-12))


def _highband_ratio_db(x: np.ndarray) -> float:
    """Energy in 8-20 kHz minus energy in 0.3-4 kHz (dB): catches band-limited or dull targets."""
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / SR)
    lo = spec[(freqs >= 300.0) & (freqs <= 4000.0)].mean() + 1e-12
    hi = spec[(freqs >= 8000.0) & (freqs <= 20000.0)].mean() + 1e-12
    return float(10.0 * np.log10(hi / lo))


def gate_pair(
    noisy: np.ndarray,
    clean: np.ndarray,
    *,
    min_sec: float = 1.5,
    min_rms_db: float = -45.0,
    max_peak: float = 0.995,
    min_hb_db: float = -48.0,
    asr_wer: float | None = None,
    max_wer: float = 0.15,
    spk_sim: float | None = None,
    min_spk_sim: float = 0.85,
) -> tuple[bool, list[str], dict]:
    """Return (accepted, refusal_reasons, metrics) for a single (noisy, teacher_clean) pair."""
    n = min(len(noisy), len(clean))
    noisy, clean = noisy[:n], clean[:n]
    reasons: list[str] = []
    dur = n / SR
    rms_in = _rms_db(noisy)
    rms_out = _rms_db(clean)
    peak_out = float(np.max(np.abs(clean))) if n else 0.0
    hb_db = _highband_ratio_db(clean) if n else -120.0

    if not (np.isfinite(noisy).all() and np.isfinite(clean).all()):
        reasons.append("non_finite")
    if dur < min_sec:
        reasons.append(f"too_short:{dur:.2f}s")
    if rms_out < min_rms_db:
        reasons.append(f"collapsed_energy:{rms_out:.1f}dBFS")
    if peak_out > max_peak:
        reasons.append(f"clipped:{peak_out:.3f}")
    if hb_db < min_hb_db:
        reasons.append(f"low_highband:{hb_db:.1f}dB")
    if asr_wer is not None and asr_wer > max_wer:
        reasons.append(f"wer_drift:{asr_wer:.3f}")
    if spk_sim is not None and spk_sim < min_spk_sim:
        reasons.append(f"spk_drift:{spk_sim:.3f}")

    metrics = {
        "dur_s": round(dur, 3),
        "rms_in_db": round(rms_in, 2),
        "rms_out_db": round(rms_out, 2),
        "peak_out": round(peak_out, 4),
        "hb_db": round(hb_db, 2),
        "asr_wer": None if asr_wer is None else round(float(asr_wer), 4),
        "spk_sim": None if spk_sim is None else round(float(spk_sim), 4),
    }
    return len(reasons) == 0, reasons, metrics


def _load_neural_scorers(device: str):
    import torch
    from jiwer import wer
    from transformers import AutoFeatureExtractor, WavLMForXVector, pipeline
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer

    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        device=0 if device == "cuda" else -1,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    norm = BasicTextNormalizer()
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    sv = (
        WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
        .to(device)
        .eval()
    )

    def score_pair(noisy_48k: np.ndarray, clean_48k: np.ndarray) -> tuple[float, float]:
        n16 = resample_poly(noisy_48k, 1, 3).astype(np.float32)
        c16 = resample_poly(clean_48k, 1, 3).astype(np.float32)
        t_in = norm(asr({"array": n16, "sampling_rate": 16_000})["text"])
        t_out = norm(asr({"array": c16, "sampling_rate": 16_000})["text"])
        w = (
            0.0
            if not t_in and not t_out
            else (1.0 if not t_in else float(wer(t_in, t_out)))
        )
        with torch.no_grad():
            bi = fe(n16, sampling_rate=16_000, return_tensors="pt").to(device)
            bo = fe(c16, sampling_rate=16_000, return_tensors="pt").to(device)
            ei = sv(**bi).embeddings
            eo = sv(**bo).embeddings
            sim = float(torch.nn.functional.cosine_similarity(ei, eo).item())
        return w, sim

    return score_pair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--noisy-dir", required=True, help="directory of degraded input clips"
    )
    ap.add_argument(
        "--teacher-dir",
        required=True,
        help="directory of matching teacher-restored 48 kHz clips",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="output dir (writes noisy/, clean/, manifest.jsonl)",
    )
    ap.add_argument(
        "--max-wer",
        type=float,
        default=0.15,
        help="max Whisper WER drift between input & teacher",
    )
    ap.add_argument(
        "--min-spk-sim",
        type=float,
        default=0.85,
        help="min WavLM speaker cosine similarity",
    )
    ap.add_argument(
        "--min-hb-db",
        type=float,
        default=-48.0,
        help="min 8-20 kHz vs 0.3-4 kHz energy ratio (dB)",
    )
    ap.add_argument(
        "--no-neural-gates",
        action="store_true",
        help="skip Whisper/WavLM checks (acoustic gates only)",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--limit", type=int, default=0, help="optional max clips to process"
    )
    args = ap.parse_args()

    noisy_root = Path(args.noisy_dir).expanduser()
    teacher_root = Path(args.teacher_dir).expanduser()
    out_root = Path(args.out).expanduser()
    (out_root / "noisy").mkdir(parents=True, exist_ok=True)
    (out_root / "clean").mkdir(parents=True, exist_ok=True)

    scorer = None if args.no_neural_gates else _load_neural_scorers(args.device)

    files = sorted(
        p
        for p in noisy_root.rglob("*")
        if p.is_file() and p.suffix.lower() in (".wav", ".flac")
    )
    if args.limit > 0:
        files = files[: args.limit]

    accepted = rejected = 0
    manifest = out_root / "manifest.jsonl"
    with open(manifest, "w") as mf:
        for idx, np_path in enumerate(files):
            rel = np_path.relative_to(noisy_root)
            tp_path = teacher_root / rel
            if not tp_path.exists():
                tp_path = teacher_root / rel.with_suffix(".wav")
            if not tp_path.exists():
                tp_path = teacher_root / rel.with_suffix(".flac")
            if not tp_path.exists():
                continue

            noisy = _read_mono_48k(np_path)
            clean = _read_mono_48k(tp_path)
            n = min(len(noisy), len(clean))
            noisy, clean = noisy[:n], clean[:n]

            asr_wer = spk_sim = None
            if scorer is not None and n >= SR:
                asr_wer, spk_sim = scorer(noisy, clean)

            ok, reasons, metrics = gate_pair(
                noisy,
                clean,
                min_hb_db=args.min_hb_db,
                asr_wer=asr_wer,
                max_wer=args.max_wer,
                spk_sim=spk_sim,
                min_spk_sim=args.min_spk_sim,
            )
            stem = "_".join(rel.with_suffix("").parts)
            row = {"id": stem, "ok": ok, "reasons": reasons} | metrics
            mf.write(json.dumps(row) + "\n")
            if ok:
                sf.write(
                    out_root / "noisy" / f"{stem}.flac", noisy, SR, subtype="PCM_24"
                )
                sf.write(
                    out_root / "clean" / f"{stem}.flac", clean, SR, subtype="PCM_24"
                )
                accepted += 1
            else:
                rejected += 1
            if (idx + 1) % 100 == 0:
                print(
                    f"[{idx + 1}/{len(files)}] accepted={accepted} rejected={rejected}",
                    flush=True,
                )

    print(f"Done: accepted={accepted}, rejected={rejected} -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
