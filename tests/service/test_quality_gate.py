import pytest

from src.contracts import ErrorCode, InferenceFeature
from src.imagequality.assess import measure
from src.imagequality.thresholds import Thresholds, ThresholdsNotCalibrated
from src.service.errors.exceptions import ModelServiceError
from src.service.runtime.quality_gate import QualityGate
from tests.quality.support import blurred_image, dark_image, encode, noisy_image

LONG_SIDE = 200
MIN_BRIGHTNESS = 40.0


@pytest.fixture
def gate():
    sharp = measure(encode(noisy_image()), LONG_SIDE).sharpness
    blur = measure(encode(blurred_image()), LONG_SIDE).sharpness
    thresholds = Thresholds(
        long_side=LONG_SIDE,
        min_sharpness=(sharp + blur) / 2,
        min_brightness=MIN_BRIGHTNESS,
    )

    return QualityGate(thresholds)


def test_sharp_image_passes(gate):
    gate.check(InferenceFeature.OCR, encode(noisy_image()))


@pytest.mark.parametrize(
    ("image", "code"),
    [
        (blurred_image(), ErrorCode.IMAGE_BLURRY),
        (dark_image(), ErrorCode.IMAGE_TOO_DARK),
    ],
)
def test_bad_image_is_rejected(gate, image, code):
    with pytest.raises(ModelServiceError) as caught:
        gate.check(InferenceFeature.OCR, encode(image))

    assert caught.value.error_code is code


def test_feature_outside_gate_is_not_checked(gate):
    gate.check(InferenceFeature.COLOR_DETECTION, encode(dark_image()))


def test_undecodable_bytes_are_unsupported_format(gate):
    with pytest.raises(ModelServiceError) as caught:
        gate.check(InferenceFeature.OCR, b"not an image")

    assert caught.value.error_code is ErrorCode.UNSUPPORTED_IMAGE_FORMAT


def test_uncalibrated_thresholds_are_rejected():
    with pytest.raises(ThresholdsNotCalibrated):
        QualityGate(Thresholds())
