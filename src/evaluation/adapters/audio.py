from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


def load_mono(path: Path, target_rate: int) -> np.ndarray:
    """Read a wav file as one channel of float samples at the wanted sample rate."""
    import soundfile  # imported here so other models do not need it

    samples, rate = soundfile.read(str(path), dtype="float32")

    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    if rate != target_rate:
        divisor = gcd(rate, target_rate)
        samples = resample_poly(samples, target_rate // divisor, rate // divisor)

    return samples.astype("float32")
