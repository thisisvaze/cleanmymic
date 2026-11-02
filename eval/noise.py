"""Background-noise meter: how far below the voice the noise in the pauses sits.

    uv run eval/noise.py file.wav [...]

Voice = loudest 5% of 20 ms frames, pauses = quietest 10%. More negative is cleaner.
Split by band: <300 Hz (rumble, fans, hum), 300 Hz-4 kHz (room, babble), >4 kHz (hiss).
"""
import sys

import numpy as np
import soundfile as sf
from scipy.signal import stft

BANDS = (0, 300), (300, 4000), (4000, 1e9)


def noise_db(x, sr):
    """(total, rumble, mid, hiss) pause level in dB relative to the voice."""
    f, _, Z = stft(x, sr, nperseg=sr // 50, noverlap=0)
    P = np.abs(Z) ** 2
    frames = 10 * np.log10(P.sum(0) + 1e-12)
    pauses = frames <= np.percentile(frames, 10)
    voice = np.percentile(frames, 95)
    db = lambda p: 10 * np.log10(p[:, pauses].sum(0).mean() + 1e-12) - voice  # noqa: E731
    return db(P), *(db(P[(f >= lo) & (f < hi)]) for lo, hi in BANDS)


if __name__ == "__main__":
    print(f"{'file':40} {'noise':>6}  {'rumble':>6} {'mid':>6} {'hiss':>6}  (dB below voice)")
    for p in sys.argv[1:]:
        x, sr = sf.read(p, dtype="float32", always_2d=True)
        print(f"{p.rsplit('/', 1)[-1]:40} " + "{:6.1f}  {:6.1f} {:6.1f} {:6.1f}".format(*noise_db(x.mean(1), sr)))
