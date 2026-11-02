![cleanmymic — private, offline studio-quality mic for Apple silicon](docs/images/cover.webp)

# CleanMyMic

Experimental microphone cleanup for Mac, processed locally with Core ML. Choose CleanMyMic in your call app, then switch noise removal on for cleaned audio or off for your original voice. Your physical microphone is chosen automatically.

**v0.1.0-alpha.1 — Experimental preview.** One installer bundles the app, virtual microphone, and Base model. [Download from Releases →](https://github.com/thisisvaze/cleanmymic/releases/latest) · [Release notes →](docs/releases/v0.1.0-alpha.1.md)

**Known issue:** intermittent voice loss has been reported in a noisy café call using AirPods Pro and Google Meet, while background audio remained audible. The cause is unresolved. Use a physical microphone directly in your call app for important calls. [Setup and troubleshooting →](INSTALL.md#voice-cuts-out)

## Install

Requires an **Apple Silicon Mac with macOS 15 or later**.

### Ask your agent

Paste this into Codex, Claude Code, or another coding agent:

```text
Help me install CleanMyMic from https://github.com/thisisvaze/cleanmymic.
Read INSTALL.md first and check my Mac's compatibility and the release status.
Use the documented installation path. If a required installer or model is
unavailable, tell me exactly what is missing before making system changes.
```

## Hear it

Two 10-second examples with stronger background noise, from scripted calls recorded in real environments. No added noise; the same playback gain is applied to every version of a clip. Open the WAV links to compare at the same volume.

| Example | Original | CleanMyMic Base |
|---|---|---|
| Office · overlapping voices | [Before](docs/audio/work-raw.wav?raw=true) | [After](docs/audio/work-base.wav?raw=true) |
| Car · cabin noise | [Before](docs/audio/phone_calls-raw.wav?raw=true) | [After](docs/audio/phone_calls-base.wav?raw=true) |

[All examples, DeepFilterNet3 and Krisp comparisons, and sources →](docs/audio/README.md) · Audio © Krisp Technologies · CC BY-NC 4.0.

## Development results

CleanMyMic Base improves transcription on some development call sets and worsens speech retention on others. Neither average scores nor processing speed establish reliable live-call behavior. [Quality results, Mac measurements, and limits →](docs/benchmarks.md)

## Code

[`mac/`](mac/) contains the app and driver build scripts. [`train/`](train/) contains the model and Core ML export code. [`eval/`](eval/) contains recording and comparison tools.

[Build, setup, and first use →](INSTALL.md) · [Contributing and local checks →](CONTRIBUTING.md) · [Release readiness →](docs/releasing.md#readiness)

## License

Original code is [MIT licensed](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) for dependencies and audio credits.
