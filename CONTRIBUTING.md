# Contributing

Use an Apple Silicon Mac with macOS 15+, Xcode (Swift 6.2+), and optionally Python 3.12+ with [uv](https://docs.astral.sh/uv/) if working on Core ML model export.

## Local checks

From the repository root:

```sh
swift build -c release --package-path mac
swift test --package-path mac
python3 mac/installer/verify.py
swiftc -O mac/live/main.swift mac/Sources/CleanMyMic/Enhancer.swift mac/Sources/CleanMyMic/StudioPolish.swift -o /tmp/cleanmymic-live-check
uv sync --locked
uv run --locked python -m compileall -q model
OMP_NUM_THREADS=2 uv run --locked model/dfn3_studio.py
git diff --check
```

With a Core ML model package available locally:

```sh
./mac/live.sh models/CleanMyMicStudio.mlpackage --check-bypass
```

For capture or routing changes, also verify an installed local build on real hardware: idle must release the physical mic with cleanup on or off, two apps reading from CleanMyMic must keep capture active until both stop, and toggling cleanup on/off mid-call must not drop audio.

## Pull requests

- Work on a feature branch or fork and open a draft PR against `main`.
- Describe the user-visible change, checks run, and anything not verified.
- Keep commits focused and retain upstream notices and licenses.

Original contributions follow the [MIT license](LICENSE). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attributions.
