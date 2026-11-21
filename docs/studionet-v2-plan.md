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

| Preset (`--size`) | Arch | Precision | Total Params | `.mlpackage` | M5 Pro `CPU_ONLY` | M5 Pro `CPU+NE` | M5 Pro `ALL` | Target Mac tier |
|---|---|---|---:|---:|---:|---:|---:|---|
| `pro` *(recommended fast path)* | `dfn3_studio_v2` | `mixed` | **4.63M** (`2.13M DFN3 + 1.84M S1 + 0.66M Ref`, `spk=0`) | **10.6 MB** | **1.597 ms** | **1.649 ms** | **1.511 ms** (RTF 0.151) | M1 / M3 / M4 / M5 Pro |
| `lite` | `studio_v2` | `mixed` | **3.01M** (`2.76M + 0.25M`, `spk=256`) | **6.8 MB** | **1.128 ms** | **1.138 ms** (RTF 0.114) | **1.146 ms** | Base M1 / M2 / distilled low-power |
| `pro` | `studio_v2` | `mixed` | **6.35M** (`5.67M + 0.68M`, `spk=256`) | **14.4 MB** | **1.916 ms** | **1.905 ms** (RTF 0.191) | **2.107 ms** | M3 / M4 / M5 Pro |
| `pro` *(exact FP32 ref)* | `studio_v2` | `fp32` | **6.35M** (`5.67M + 0.68M`, `spk=256`) | **25.8 MB** | **2.278 ms** | **2.059 ms** (RTF 0.206) | **2.034 ms** | M3 / M4 / M5 Pro |
| `max` *(teacher / flagship)* | `studio_v2` | `mixed` | **9.01M** (`8.33M + 0.68M`, `spk=256`) | **19.8 MB** | **2.505 ms** | **2.508 ms** (RTF 0.251) | **2.466 ms** | M4 Pro / M5 Pro / online teacher for `lite` |

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

## Measured H100 Pilot Results — `DFN3StudioV2-Pro` (Nov 21, 2026)

Trained on `cmm-fi-h100` (NVIDIA H100 80 GB, now verified `STOPPED`) from `hold-10000.pt` + 6-block full-band DualPathTF (`4.76M` Stage 1, 8,303 steps + 2,000-step loudness/pause/high-band fix tune) + frozen-Stage-1 3-block GAN refiner (`5.42M` total, `0.66M` trainable in Stage 2: `dfn3v2-s2` 6,215 steps; `dfn3v2-s2b` 3,027 steps with `--rec-weight 0.20` and 2 s GAN crops), using `2,166` WavLM-gated (`mean_spk_sim=0.9924`) `NVIDIA-NemotronLabs-RE-USE` (OpenMDW-1.1) teacher pairs (`990` RealMAN train rooms + `1,176` SLR83 accents) plus `hifitts_clean` (65% Stage-2 share), `vctk1`, `vocalset`, and `1,320` Surrey mic IRs.

### Exported trained checkpoint (`dfn3v2-s2b`, FP32 Core ML on Apple M5 Pro)
- **Model size:** `5.42M` params (`18 MB` `.mlpackage`, FP32).
- **Parity & Swift check:** **`0.0000`** worst relative error over 200 sequential frames + silence/sparse finite + Swift `Enhancer --check-bypass` **PASS**.
- **Speed (isolated Core ML on M5 Pro):** **`1.488 ms`** `CPU_ONLY` / **`1.502 ms`** `CPU_AND_NE` / **`1.501 ms`** `ALL` per 10 ms hop (`RTF 0.149–0.150`). *(Note: `--precision mixed` failed the 200-frame relative-error gate on trained noise-suppressing weights; use default `--precision fp32` for trained `DFN3StudioV2` checkpoints.)*

### Multi-device benchmark vs. shipped `DFN3Studio-v2` (Whisper large-v3-turbo + Audiobox PQ)

| Benchmark | Metric | Shipped `DFN3Studio-v2` | `DFN3StudioV2-s2b` (trained) |
|---|---|---:|---:|
| **15 val non-speech clips** (`dns`/`fsd50k`/`cochlscene`) | Attenuation | 52.2 dB | **76.7 dB** (+24.5 dB) |
| **15 val non-speech clips** | Whisper words (> −70 dBFS) | 231 | **3** (−98.7%) |
| **DAPS** (matched 23 real iPad/iPhone rooms) | WER | 7.0% | **6.5%** (`s2`: **6.3%**) |
| **DAPS** (matched 23) | Tone error vs. studio | 2.21 dB | **2.10 dB** |
| **DAPS** (matched 23) | Audiobox PQ | **5.87** | 5.82 |
| **Unseen RealMAN `Classroom2`** (n=16, −24 dBFS preamp) | Audiobox PQ | 5.66 | **5.71** |
| **Unseen RealMAN `OfficeRoom3`** (n=16, −24 dBFS preamp) | Audiobox PQ | 4.51 | **4.56** |
| **Krisp `krisp-work`** (full 40 real calls) | WER / PQ / bg voices | — | **10.8%** / **5.71** / **−43.8 dB** |
| **Krisp `krisp-call`** (full 40 call-center calls) | WER / PQ / bg voices | — | **13.0%** / **6.58** / **−45.7 dB** |
| **Krisp `krisp-phone`** (full 40 phone calls) | WER / PQ / bg voices | — | **4.5%** (`s2`) / **4.37** / **−52.9 dB** |
| **Krisp `krisp-work`** (matched 12-clip subset) | WER / bg voices | **15.0%** / −22.2 dB | 16.3% / **−24.4 dB** |

**Key takeaway on the `PQ ~5.85 → 6.7+` ceiling:** `DFN3StudioV2-s2b` delivered major gains in non-speech hallucination suppression (`231 → 3` words, `76.7 dB` attenuation), DAPS tone (`2.10 dB`), DAPS WER (`7.0% → 6.5%`), unseen RealMAN rooms, and multi-talker bg-voice suppression (`−43.8` to `−52.9 dB` across all three Krisp call sets), while staying fast at `1.49 ms/hop` in exact FP32. However, keeping DFN3's 32-band ERB / 4.8 kHz suppressor at the front still caps DAPS PQ near `5.85` (especially on heavy outdoor noise like `balcony1`, where Base itself drops to `PQ 4.17` vs. offline `NVIDIA-NemotronLabs-RE-USE` `6.31`). Reaching `PQ 6.7+` requires training the full-band `StudioNetV2-Pro` backbone (no DFN3 front-end) for a full 50k–60k-step run (~8–10 h on H100) with teacher distillation.

**Not yet verified:** Blind human listening on `dfn3v2-s2b` vs. shipped `DFN3Studio-v2`, M1/M3/M4 Pro hardware timings, and live macOS HAL call-app testing; `/Applications/CleanMyMic.app` has not been replaced.
