from src.contracts import ErrorCode, InferenceFeature
from src.imagequality.assess import Quality, assess
from src.imagequality.metrics import ImageDecodeError
from src.imagequality.thresholds import Thresholds

from ..errors.exceptions import ModelServiceError

GATED_FEATURES = frozenset({InferenceFeature.OCR})
REJECTIONS = {
    Quality.BLUR: ErrorCode.IMAGE_BLURRY,
    Quality.DARK: ErrorCode.IMAGE_TOO_DARK,
}


class QualityGate:
    """reject blurry or dark images before the model runs and it is not wired yet"""

    def __init__(
        self,
        thresholds: Thresholds,
        features: frozenset[InferenceFeature] = GATED_FEATURES,
    ):
        thresholds.require_ready()
        self._thresholds = thresholds
        self._features = features

    def check(self, feature: InferenceFeature, image: bytes) -> None:
        if feature not in self._features:
            return

        try:
            quality = assess(image, self._thresholds).quality
        except ImageDecodeError:
            raise ModelServiceError(ErrorCode.UNSUPPORTED_IMAGE_FORMAT) from None

        if quality in REJECTIONS:
            raise ModelServiceError(REJECTIONS[quality])
