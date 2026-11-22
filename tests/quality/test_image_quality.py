import numpy as np
import pytest

from src.imagequality.assess import Quality, assess, classify, measure
from src.imagequality.metrics import ImageDecodeError, decode_gray, normalize
from src.imagequality.thresholds import (
    Thresholds,
    ThresholdsNotCalibrated,
    load_thresholds,
    save_thresholds,
)
from tests.quality.support import (
    SIDE,
    blurred_image,
    dark_image,
    encode,
    noisy_image,
)

LONG_SIDE = 200
MIN_BRIGHTNESS = 40.0


def _thresholds():
    sharp = measure(encode(noisy_image()), LONG_SIDE).sharpness
    blur = measure(encode(blurred_image()), LONG_SIDE).sharpness

    return Thresholds(
        long_side=LONG_SIDE,
        min_sharpness=(sharp + blur) / 2,
        min_brightness=MIN_BRIGHTNESS,
    )


@pytest.mark.parametrize(
    ("image", "expected"),
    [
        (noisy_image(), Quality.SHARP),
        (blurred_image(), Quality.BLUR),
        (dark_image(), Quality.DARK),
    ],
)
def test_assess_labels_each_case(image, expected):
    assert assess(encode(image), _thresholds()).quality is expected


def test_blur_measures_lower_than_sharp():
    sharp = measure(encode(noisy_image()), LONG_SIDE)
    blur = measure(encode(blurred_image()), LONG_SIDE)

    assert blur.sharpness < sharp.sharpness


def test_uncalibrated_thresholds_refuse_to_classify():
    reading = measure(encode(noisy_image()), LONG_SIDE)

    with pytest.raises(ThresholdsNotCalibrated):
        classify(reading, Thresholds())


def test_thresholds_round_trip(tmp_path):
    path = tmp_path / "t.json"
    save_thresholds(path, _thresholds())

    assert load_thresholds(path) == _thresholds()


def test_garbage_bytes_fail_to_decode():
    with pytest.raises(ImageDecodeError):
        decode_gray(b"not an image")


def test_normalize_only_shrinks():
    gray = np.zeros((SIDE, SIDE), dtype=np.uint8)

    assert max(normalize(gray, LONG_SIDE).shape) == LONG_SIDE
    assert normalize(gray, SIDE * 2).shape == gray.shape
