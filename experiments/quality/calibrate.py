import numpy as np

from src.imagequality.assess import Quality
from src.imagequality.thresholds import Thresholds

from .constants import BLUR_BETA, DARK_BETA
from .samples import Sample
from .search import best_cutoff


def calibrate(samples: list[Sample], dataset: str) -> Thresholds:
    """dark first on brightness then blur on sharpness of the lit images"""
    min_brightness = best_cutoff(
        np.array([s.reading.brightness for s in samples]),
        np.array([s.label is Quality.DARK for s in samples]),
        DARK_BETA,
    )
    lit = [s for s in samples if s.reading.brightness >= min_brightness]
    min_sharpness = best_cutoff(
        np.array([s.reading.sharpness for s in lit]),
        np.array([s.label is Quality.BLUR for s in lit]),
        BLUR_BETA,
    )

    return Thresholds(
        min_sharpness=min_sharpness,
        min_brightness=min_brightness,
        calibrated_on=dataset,
    )
