# StudioNet v2: Full-Band Causal Two-Stage Studio Restoration

## Why StudioNet v2

`DFN3Studio-v2` (3.50M params, `0.54 ms`/10 ms frame) still bottlenecks through DeepFilterNet3's 32-band ERB + 0–4.8 kHz core and unfroze Stage 1 during GAN fine-tuning (which lifted DAPS PQ `5.76 → 5.87` but regressed DAPS WER `5.1% → 7.0%` and left 10 Whisper words on 15 non-speech clips).

StudioNet v2 applies four findings from 2026 universal speech enhancement research (NVIDIA RE-USE / `arXiv:2603.02641`, `RT-SEMamba`, personalized restoration `arXiv:2611.22620`) directly to CleanMyMic's Core ML streaming contract:

1. **Full-band causal dual-path backbone (`StudioNet`)**: operates across all 481 STFT bins (0–24 kHz) with frequency self-attention + per-band causal GRUs + global 40 s speaker-context GRU.
2. **Frozen Stage-1 Regression + Zero-Init Stage-2 Residual GAN Refiner (`StudioNetV2`)**: Stage 1 trains to convergence for maximum word & speaker fidelity, then freezes (`freeze_stage1()`) while Stage 2 (`StudioRefiner`, seeing both Stage-1 output `Y1` and raw noisy `X_t` plus Stage-1 global context `g_out`) restores over-smoothed 8–24 kHz air and vocal harmonics without corrupting Stage-1 fidelity.
3. **Commercial-clean teacher distillation (`NVIDIA-NemotronLabs-RE-USE`, OpenMDW-1.1)** + **online student distillation (`--teacher-ckpt`)**: distill studio polish from the 9.6M bi-directional Mamba teacher (gated on Whisper WER + WavLM speaker similarity in `train/prep_teacher.py`), and optionally distill `Pro` → `Lite` for older/base Apple Silicon Macs.
4. **Optional speaker FiLM (`--spk-dim 256`) + harder anti-hallucination & codec degradations**: zero-init FiLM defaults to exact zero-enrollment mode (drop-in with `Enhancer.swift`) while keeping the slot ready for 5-second voice enrollment; `studio_data.py` adds direct-path RIR alignment, Bluetooth HFP/LC3/VoIP/upstream-suppressor artifacts, and distant-chatter/event non-speech zero-targets with an explicit −55 dB floor loss.

## Core ML Sizing Benchmarks (Apple M5 Pro, macOS 27, 48 kHz, 10 ms hop)

Measured November 20, 2026 via `train/export.py` (`--precision mixed`: FP16 on `linear`/`matmul`/`gru`, FP32 on convolutions and spectral math; plus `fp32` reference for `pro`). Every exported `.mlpackage` passed 200-frame PyTorch-vs-Core ML parity, digital-silence & sparse-tone finiteness checks, and `mac/live/main.swift --check-bypass` in the real Swift `Enhancer.swift` engine.

| Preset (`--size`) | Precision | Total Params (`S1 + Refiner`, `spk_dim=256`) | `.mlpackage` | M5 Pro `CPU_ONLY` | M5 Pro `CPU+NE` | M5 Pro `ALL` | Worst Parity (200 fr) | Target Mac tier |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `lite` | `mixed` | **3.01M** (`2.76M + 0.25M`) | **6.8 MB** | **1.128 ms** | **1.138 ms** (RTF 0.114) | **1.146 ms** | **0.0173** | Base M1 / M2 / distilled low-power |
| `pro` *(default target)* | `mixed` | **6.35M** (`5.67M + 0.68M`) | **14.4 MB** | **1.916 ms** | **1.905 ms** (RTF 0.191) | **2.107 ms** | **0.0228** | M3 / M4 / M5 Pro |
| `pro` *(exact FP32 ref)* | `fp32` | **6.35M** (`5.67M + 0.68M`) | **25.8 MB** | **2.278 ms** | **2.059 ms** (RTF 0.206) | **2.034 ms** | **0.0000** | M3 / M4 / M5 Pro |
| `max` *(teacher / flagship)* | `mixed` | **9.01M** (`8.33M + 0.68M`) | **19.8 MB** | **2.505 ms** | **2.508 ms** (RTF 0.251) | **2.466 ms** | **0.0168** | M4 Pro / M5 Pro / online teacher for `lite` |

*(Measured on Apple M5 Pro with isolated Core ML `predict` loops; M1/M3/M4 Pro timings and live HAL audio-thread p99 latency have not yet been measured on hardware.)*

## Training & Evaluation Runbook (H100/H200)

### 0. Prepare verified teacher pairs (optional, commercial-clean OpenMDW-1.1)
Run `nvidia/NVIDIA-NemotronLabs-RE-USE` (or `NVIDIA-NemotronLabs-RE-USE-laughter`) over diverse degraded speech (e.g., RealMAN + filtered HiFiTTS-2 + DNS5), then enforce fidelity gates:

```sh
uv run --group eval train/prep_teacher.py \
  --noisy-dir ~/data/real_noisy \
  --teacher-dir ~/data/nemotron_reuse_48k \
  --out ~/experiments/studionet-v2/teacher_pairs \
  --max-wer 0.15 --min-spk-sim 0.85
```

### 1. Stage 1 — High-fidelity full-band regression (`StudioNet-Pro`)
```sh
uv run train/train_studio.py ~/data ~/experiments/studionet-v2/s1_pro \
  --arch studio --size pro --steps 60000 --batch 20 --seg-sec 4 \
  --mic-irs ~/experiments/studio-20261119/data/surrey_mics_48k.npy \
  --real-pairs ~/experiments/realman-20261119/pairs/train \
  --teacher-pairs ~/experiments/studionet-v2/teacher_pairs \
  --extra-clean ~/experiments/studio-20261119/data/hifitts_clean \
  --wavlm-weight 3.0 --noise-only-prob 0.12
```

### 2. Stage 2 — Frozen-Stage-1 Residual GAN Refiner (`StudioNetV2-Pro`)
Stage 1 freezes automatically under `--arch studio_v2 --gan`, so adversarial training sharpens 8–24 kHz air and vocal timbre without regressing WER or pause hold:

```sh
uv run train/train_studio.py ~/data ~/experiments/studionet-v2/s2_pro \
  --arch studio_v2 --size pro --gan --steps 20000 --batch 16 --seg-sec 4 \
  --init ~/experiments/studionet-v2/s1_pro/best.pt \
  --mic-irs ~/experiments/studio-20261119/data/surrey_mics_48k.npy \
  --real-pairs ~/experiments/realman-20261119/pairs/train \
  --teacher-pairs ~/experiments/studionet-v2/teacher_pairs \
  --extra-clean ~/experiments/studio-20261119/data/hifitts_clean \
  --wavlm-weight 3.0 --noise-only-prob 0.12
```

### 3. Optional Stage 3 — Distill `Pro` → `Lite` for M1 / ultra-low power
```sh
uv run train/train_studio.py ~/data ~/experiments/studionet-v2/s1_lite_kd \
  --arch studio --size lite --steps 40000 --batch 24 --seg-sec 4 \
  --teacher-ckpt ~/experiments/studionet-v2/s2_pro/best.pt --distill-weight 0.5 \
  --mic-irs ~/experiments/studio-20261119/data/surrey_mics_48k.npy \
  --real-pairs ~/experiments/realman-20261119/pairs/train \
  --extra-clean ~/experiments/studio-20261119/data/hifitts_clean \
  --wavlm-weight 2.0 --noise-only-prob 0.12
```

### 4. Export to Core ML (Mac) & evaluate across public multi-device sets
```sh
uv run train/export.py ~/experiments/studionet-v2/s2_pro/best.pt \
  models/StudioNetV2-Pro.mlpackage --precision mixed

uv run --group eval eval/score.py ~/experiments/studionet-v2/s2_pro/best.pt \
  --no-voice --data ~/data \
  --sets daps,drvctk,dns5,krisp-work,krisp-call,krisp-phone
```

**Promotion gates (vs. current `DFN3Studio-v2`):**
- DAPS WER `< 4.5%` (down from `7.0%`) and PQ `≥ 6.5` (up from `5.87`).
- Multi-talker AirPods `krisp-work` WER `≤ 14.5%` and bg-voice suppression `≥ 32 dB`.
- Non-speech clips: **0 Whisper hallucinated words** (down from 10).
- Clean studio input nearly unchanged; Core ML worst parity `< 0.01` and M5 Pro frame time `< 2.5 ms`.
