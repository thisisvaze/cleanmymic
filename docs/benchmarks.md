# Development benchmarks

Quality scored October 31, 2026 for **CleanMyMic Base**, the app's default checkpoint (previously `Hold-10k`; `hold-10000-novoice` in the evidence files). Mac timing updated November 1. These are development results, not a ranking of Mac products.

**Live-call failure reported November 1:** intermittent loss of the caller's voice while café background audio/music remained audible, using AirPods Pro through CleanMyMic into Google Meet. Meet noise cancellation may have been enabled; the cause is not isolated. The tests below do not reproduce or clear this failure. See [troubleshooting](../INSTALL.md#voice-cuts-out).

## Word retention

Mean per-clip word error, capped at 100%; lower is better, best in each row in bold. Each row uses the same 40 clips for every system.

| Set | Raw input | CleanMyMic Base | Krisp VI 2.5 HD | DeepFilterNet3 |
|---|---:|---:|---:|---:|
| Work calls | 26.1% | **9.7%** | 10.3% | 20.6% |
| Call center | 20.3% | **13.1%** | 14.8% | 19.8% |
| Phone calls | **3.8%** | 4.1% | 4.3% | 6.3% |

[Per-clip numeric results](benchmark-data/quality-pilot-20261031.csv) and [provenance, corpus WER, and uncertainty](benchmark-data/quality-pilot-20261031.json) cover all five systems and eight sets: 1,400 system/clip results. Original transcripts and audio are not included.

Base minus Krisp work-call WER is −0.64 percentage points, with a paired speaker-bootstrap 95% interval of −2.39 to +1.19. That small advantage is uncertain. The call-center interval is −4.07 to −0.07 points, but includes only seven speakers. Phone's interval also includes zero. These comparisons do not establish overall superiority.

## Background voices

Suppression during annotated secondary-speaker-only regions, averaged per clip in dB; higher is better, best in each row in bold. This table applies the same **48 kHz, 16-bit PCM export** to every system's output. Existing quantization and bandwidth limits in competitor files cannot be undone.

| Set | Qualifying clips | CleanMyMic Base | Krisp VI 2.5 HD | DeepFilterNet3 |
|---|---:|---:|---:|---:|
| Work calls | 40 | **32.3 dB** | 30.9 dB | 14.8 dB |
| Call center | 39 | **29.6 dB** | 29.4 dB | 18.6 dB |
| Phone calls | **4** | 5.9 dB | 17.9 dB | **37.4 dB** |

Phone suppression remains a weakness, with particularly little evidence. Suppression alone rewards silencing everything; read it alongside word retention. These regions do not measure separation while both people speak.

Export format matters near silence: Base's native work/call suppression is 36.7/35.5 dB, compared with 32.3/29.6 dB after the common PCM16 export. Both measurements are retained in the evidence files; evaluation WAVs preserve float output for reanalysis.

## Speech preservation

Base WER is 6.52% on DAPS and 9.27% on DR-VCTK, worse than raw input at 3.09% and 5.94%. Those references are ASR transcripts of clean audio, not human transcripts. Capped means can hide large failures; the evidence also includes uncapped corpus WER and individual error counts.

The evidence includes two 2,000-step fine-tunes from Base: continuation and corrected near/far training. Neither consistently improved all sets, so neither replaced Base. Training settings and per-checkpoint results remain in the provenance file.

A [separate demo audit](benchmark-data/demo-quality-20261031.json) found higher predicted intelligibility (eSTOI) for DeepFilterNet3 than Base on all three earlier prepared noise mixes. Lowering Base to 70% strength did not consistently improve that score. Those clips use one narrator; they do not establish a general ranking, and eSTOI does not measure listening preference. Base is not established as the best listening-quality checkpoint. The [real-world call comparisons](audio/README.md) are available for direct review.

## Method and limits

- CleanMyMic: no enrollment, full strength, no input gain; PyTorch inference. The Base checkpoint SHA-256 is `20e1b4792ec7698473f38815f2d7a22b02ae0a2a98f882048373514e4c3a6c03`.
- Krisp: [published VI 2.5 HD outputs](https://huggingface.co/datasets/Krisp-AI/VoiceIsolation-Benchmark-Dataset-Processed) for its [Voice Isolation Benchmark](https://huggingface.co/datasets/Krisp-AI/VoiceIsolation-Benchmark-Dataset). This does not benchmark the current Krisp Mac app.
- DeepFilterNet3: stock 0.5.6 CLI defaults and released model-120 weights; existing 48 kHz PCM16 outputs. Version and checkpoint hash are recorded.
- ASR: Whisper large-v3-turbo, identical settings across systems. Reference transcripts matched in every fresh run. Repeated raw-input decoding varied on a few clips; the reported raw-input row consistently uses the Base pass. Counts and transcripts were checked before publishing numeric results.
- Quality predictions in the data use Audiobox Aesthetics, not human listening scores.
- These 40-clip samples already informed development. This was one run per training recipe, not a repeated-seed study. A fresh independent holdout is still needed.

## Mac implementation check

A separate [parity check](benchmark-data/hold-coreml-parity.json) used the installed Base Core ML model and the app's shared audio processor on two recordings, with added quiet speech and silence. Core ML and PyTorch agreed above 112 dB numerical SNR overall and 71 dB in the quiet sections. File processing on an M5 Pro took 0.068–0.075 seconds per second of audio, excluding initialization and device routing. The record documents the file reader's short tail and comparison limits.

This does not measure live-call latency, battery life, long-session reliability, or listening preference. A same-device comparison against [Apple Voice Isolation](https://support.apple.com/en-gb/guide/mac-help/mchle82b42f0/27/mac/27), the current Krisp app, and additional open-source models remains outstanding. The supported positioning is local processing on Apple Silicon; “best on Mac” is not established.

On November 1, an isolated per-frame test of the same Base weights on an **M5 Pro, macOS 27.2** measured **0.212 ms mean processing per 10 ms frame** across 17,555 frames from three existing call-recording prefixes. Each clip's p95 was below 0.24 ms; none exceeded 10 ms. [Timing method, hashes, and per-clip results](benchmark-data/base-coreml-timing-20261101.json) identify the measured source and inputs. This single unpaced run excludes model loading, file IO, device routing, and call-app/network buffering. It is a different workload from the earlier file-processing test, not proof of a model optimization or a CPU-utilization measurement.

Model metadata specifies a 20 ms window and two 10 ms look-ahead hops: **40 ms window-plus-look-ahead latency**, before processing and device/call buffering. File output shifts by 30 ms because of overlap and padding. Neither figure is a measured end-to-end call delay. These timings establish processing headroom on this M5 Pro, not speech preservation or performance across all Apple Silicon Macs.

## Resource use

Measured November 2, 2026 on an **M5 Pro, macOS 27.2**, with the app's built-in check (`CleanMyMic --check "MacBook Pro Microphone" "CleanMyMic (for calls)"`; app build `e0df54c`, same audio pipeline as `main`). It runs live microphone audio through Base into the virtual microphone for two 4-second runs, switching cleanup on and off each second. Both runs kept real time at about 100 frames per second.

- **CPU:** 0.14 s of process CPU time over a steady 5-second window, about 3% of one core. Core ML may also schedule work on the Neural Engine, which this does not count.
- **Memory:** 43 MB resident throughout.
- **Size:** the compiled Base model is 8.4 MB; the installer with app, virtual microphone, and model is 13.6 MB.

One short run on one Mac, not an energy or battery measurement.

## Reproduce scoring

```sh
uv run --group eval eval/score.py /path/to/checkpoint.pt --no-voice --data /path/to/data --sets krisp-work,krisp-call,krisp-phone
uv run --group eval eval/score.py /path/to/processed-audio --data /path/to/data --sets krisp-work,krisp-call,krisp-phone
uv run --no-project eval/compare.py eval/outputs/system-a eval/outputs/system-b
```

Use matching input filenames from `eval/score.py --dump`, and one score file per clip and system: overlapping CSVs are otherwise counted twice. `compare.py` reports the native-output suppression metric; the common-format sensitivity calculation and its per-clip values are documented in the provenance and numeric export.
