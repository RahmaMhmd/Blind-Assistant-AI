from dataclasses import dataclass
from enum import Enum

from .metrics import brightness, decode_gray, normalize, sharpness
from .thresholds import Thresholds


class Quality(str, Enum):
    SHARP = "sharp"
    BLUR = "blur"
    DARK = "dark"


@dataclass(frozen=True)
class Reading:
    sharpness: float
    brightness: float


@dataclass(frozen=True)
class Assessment:
    quality: Quality
    reading: Reading


def measure(data: bytes, long_side: int) -> Reading:
    gray = normalize(decode_gray(data), long_side)

    return Reading(sharpness(gray), brightness(gray))


def classify(reading: Reading, thresholds: Thresholds) -> Quality:
    """dark is checked first because dark images also look blurry"""
    thresholds.require_ready()

    if reading.brightness < thresholds.min_brightness:
        return Quality.DARK
    if reading.sharpness < thresholds.min_sharpness:
        return Quality.BLUR

    return Quality.SHARP


def assess(data: bytes, thresholds: Thresholds) -> Assessment:
    reading = measure(data, thresholds.long_side)

    return Assessment(classify(reading, thresholds), reading)
