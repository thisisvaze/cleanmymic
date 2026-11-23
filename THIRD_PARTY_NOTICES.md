# Third-party notices

Original CleanMyMic code and documentation use the root [MIT license](LICENSE). Third-party components retain their original licenses below.

## DeepFilterNet

[`model/dfn3.py`](model/dfn3.py) adapts layers, features, and framing from [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet), copyright (c) 2021 Hendrik Schröter. DeepFilterNet offers MIT or Apache-2.0 licensing; this repository includes its [MIT notice](licenses/DeepFilterNet-MIT.txt).

## BlackHole virtual audio driver

[`mac/driver/build.sh`](mac/driver/build.sh) fetches [BlackHole](https://github.com/ExistentialAudio/BlackHole) at commit `574029f0d3175ccf29a826d74be4da3967b1cb92`, copyright © 2019-2026 Existential Audio Inc. The fetched driver source uses GPL-3.0; its [upstream license and notices](licenses/BlackHole.txt) are kept separately from the app's MIT license.

## Models and dependencies

Model checkpoints and Core ML packages are excluded from git. Python export dependencies are listed in [`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock) and retain their own licenses.
