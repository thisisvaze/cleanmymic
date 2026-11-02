# Third-party notices

Original CleanMyMic code and documentation use the root MIT license. Third-party components and audio retain the notices below.

## DeepFilterNet

[`train/dfn3.py`](train/dfn3.py) adapts the layers, features, and framing from [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet), copyright (c) 2021 Hendrik Schröter. DeepFilterNet offers MIT or Apache-2.0 licensing; this repository includes its [MIT notice](licenses/DeepFilterNet-MIT.txt). The source file retains attribution.

## BlackHole virtual audio driver

[`mac/driver/build.sh`](mac/driver/build.sh) fetches [BlackHole](https://github.com/ExistentialAudio/BlackHole) at commit `574029f0d3175ccf29a826d74be4da3967b1cb92`, copyright © 2019–2026 Existential Audio Inc. The fetched driver source uses GPL-3.0; its [upstream license and notices](licenses/BlackHole.txt) are retained separately from the app's MIT license.

The build script records the name, device configuration, icon, and transport changes. Driver binary releases include the applicable notices and corresponding source.

## Models and data

Checkpoints, Core ML packages, private recordings, and downloaded datasets are excluded from git.

## Other dependencies

The [audio examples](docs/audio/README.md) are excerpts of Krisp Technologies' Voice Isolation Benchmark recordings and matched processed outputs, with additional labeled CleanMyMic/DeepFilterNet processing. They retain CC BY-NC 4.0; source URLs, attribution, and modifications are documented alongside them.

Python dependencies are recorded in [`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock). They retain their own licenses.
