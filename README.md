![cleanmymic — private, offline and cleaner mic for Apple silicon](docs/images/cover.webp)

# CleanMyMic

Removes background noise and nearby voices from your microphone in Zoom, Meet, Teams, and any other app. It runs entirely on your Mac: no account, no cloud.

**[⬇ Download for Mac](https://github.com/thisisvaze/cleanmymic/releases/download/v0.1.0-alpha.1/CleanMyMic-0.1.0-alpha.1-arm64.pkg)** · Apple Silicon, macOS 15 or later · [▶ Hear samples](https://thisisvaze.github.io/cleanmymic/audio/)

## Get started

1. **Install.** Open the downloaded `.pkg` and follow the installer. It asks for your Mac's password.
2. **Restart your Mac** so the CleanMyMic microphone appears.
3. **Open CleanMyMic** from Applications and allow microphone access. It lives in your menu bar.
4. **Pick CleanMyMic as the microphone** in your call app, and turn the app's own noise suppression off or to Low.

Put on headphones and try **Mic check** in the menu bar to hear the difference. The switch at the top turns cleanup on or off; your call app keeps the same microphone either way.

**Experimental preview** · [Known issues](https://github.com/thisisvaze/cleanmymic/issues?q=is%3Aissue%20state%3Aopen%20label%3Abug) · [Release notes](docs/releases/v0.1.0-alpha.1.md) · [Setup and troubleshooting](INSTALL.md)

### Or ask your agent

Paste this into Codex, Claude Code, or another coding agent:

```text
Help me install CleanMyMic from https://github.com/thisisvaze/cleanmymic.
Read INSTALL.md first and check my Mac's compatibility and the release status.
Use the documented installation path. If a required installer or model is
unavailable, tell me exactly what is missing before making system changes.
```

## Hear it

Three 10-second clips from real office, call-center, and car recordings with strong background noise; no noise was added. Play the original, CleanMyMic Base, and Krisp or DeepFilterNet3 side by side at the same volume:

**[▶ Listen in your browser](https://thisisvaze.github.io/cleanmymic/audio/)**

Audio © Krisp Technologies · CC BY-NC 4.0 · [Sources and method →](docs/audio/README.md)

## Lightweight

| | CleanMyMic Base |
|---|---|
| Processing per 10 ms of audio | **0.21 ms** |
| Model latency | **40 ms** (20 ms window + 20 ms look-ahead) |
| CPU while in a call | **~3%** of one core |
| Memory | **~43 MB** |
| Size | **8.4 MB** model · 13.6 MB installer with app and virtual mic |

Measured on an M5 Pro with macOS 27.2. CPU excludes any Neural Engine work; latency excludes audio-device and call-app buffering. [Method →](docs/benchmarks.md#resource-use)

## Benchmarks

Word error rate on Krisp's real-world [Voice Isolation Benchmark](https://huggingface.co/datasets/Krisp-AI/VoiceIsolation-Benchmark-Dataset): mean per-clip WER from Whisper large-v3-turbo, 40 clips per set, lower is better. Measured October 31, 2026 for CleanMyMic Base.

| Set | Raw input | CleanMyMic Base | Krisp VI 2.5 HD | DeepFilterNet3 |
|---|---:|---:|---:|---:|
| Work calls | 26.1% | **9.7%** | 10.3% | 20.6% |
| Call center | 20.3% | **13.1%** | 14.8% | 19.8% |
| Phone calls | **3.8%** | 4.1% | 4.3% | 6.3% |

Best in each row is bold; on phone calls no processor beat the raw input. Krisp is its published benchmark output, not the current Krisp app; DeepFilterNet3 is stock 0.5.6. The gaps to Krisp are small and uncertain (few speakers per set), so this is not a ranking. Base removes background voices well on work and call-center calls (about 30 dB) but poorly on phone calls, and it costs some words on clean speech (6.5% vs 3.1% WER on DAPS). [Full results, uncertainty, and method →](docs/benchmarks.md)

## Code

[`mac/`](mac/) contains the app and driver build scripts. [`train/`](train/) contains the model and Core ML export code. [`eval/`](eval/) contains recording and comparison tools.

[Build, setup, and first use →](INSTALL.md) · [Contributing and local checks →](CONTRIBUTING.md) · [Release readiness →](docs/releasing.md#readiness)

## License

Original code is [MIT licensed](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) for dependencies and audio credits.
