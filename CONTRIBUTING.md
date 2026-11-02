# Contributing

Use an Apple Silicon Mac with macOS 15+, Xcode with Swift 6.2+, Python 3.12+, and [uv](https://docs.astral.sh/uv/). See [INSTALL.md](INSTALL.md) for app setup and the required models.

## Local checks

From the repository root:

```sh
uv sync --locked --python 3.12
uv run --locked python -m compileall -q train eval
OMP_NUM_THREADS=2 uv run --locked train/model.py
swift build -c release --package-path mac
swift test --package-path mac
python3 mac/installer/verify.py
swiftc -O mac/live/main.swift mac/Sources/CleanMyMic/Enhancer.swift -o /tmp/cleanmymic-live-check
swiftc -O eval/latency.swift mac/Sources/CleanMyMic/Enhancer.swift -o /tmp/cleanmymic-latency-check
git diff --check
```

CI runs these same checks on Apple Silicon. They cover streaming/offline agreement for the original model family, microphone routing policy, installer configuration, and app/tool compilation. No datasets or model downloads are needed; these checks do not establish Base quality or live audio behavior. With Base available, build and inspect the actual installer using [the release recipe](docs/releasing.md#build-the-installer).

With the Base model available locally, run `./mac/live.sh models/hold-10k.mlpackage --check-bypass`. This synthetic check opens no microphone; it checks unchanged original-voice samples, frozen model state during bypass, and a clean resume without stale speech.

For capture changes, also check an installed development build: idle must release the physical mic with noise removal either on or off; two apps using CleanMyMic must keep it active until both stop. Check quick open/close requests and switching noise removal off/on during use without changing the call app's input. These hardware checks are not covered by CI.

For scoring, install the evaluation dependencies with `uv sync --locked --group eval` and follow [the benchmark method](docs/benchmarks.md). Record the exact checkpoint, dataset, inference settings, and scoring method with any new result. Keep recordings and transcripts out of PRs unless you have permission to share them.

## Current priority: preserve the caller's voice

The [café/AirPods/Meet voice-loss report](INSTALL.md#voice-cuts-out) is unresolved. Before promoting another checkpoint, reproduce the failure and compare the physical input, local cleaned output, and received call audio. Include quiet speech, returning after pauses, competing voices/music, and actual AirPods capture; a bandwidth simulation alone does not test Bluetooth hardware or Meet. Keep call-app processing settings fixed during each comparison, and check raw/bypass alongside full cleanup. Report speech deletions and bad individual clips, not only mean noise suppression. A speed test or more raw audio mixed into the output is not evidence that this failure is fixed.

## Pull requests

- Work on a feature branch or fork and open a draft PR against `main`.
- Describe the user-visible change, checks run, and anything unverified. Include a small regression check for non-trivial logic.
- Keep changes focused and preserve upstream attribution.
- The owner, [@thisisvaze](https://github.com/thisisvaze), reviews changes and controls merging. Agents must not merge, enable auto-merge, or publish releases without explicit authorization.

[CODEOWNERS](.github/CODEOWNERS) names the reviewer; it does not enforce approval. See [release preparation](docs/releasing.md) for the current protection limits and publication steps.

Original contributions follow the [MIT license](LICENSE). Retain upstream attribution and applicable licenses; [third-party notices](THIRD_PARTY_NOTICES.md) distinguish the app code, driver, models, and datasets.
