from typing import NoReturn

from .errors import ErrorCode
from .features import FEATURE_INPUT_TYPES, InferenceFeature, InputKind
from .image_rules import DEFAULT_MAX_IMAGE_BYTES
from .image_validation import validate_image_base64
from .validation_error import ContractValidationError
from .validation_messages import Problem


def _fail(problem: str) -> NoReturn:
    raise ContractValidationError(ErrorCode.VALIDATION_FAILED, problem)


def _require_non_blank(value: str | None, missing: str, blank: str) -> str:
    if value is None or not value:
        _fail(missing)
    if not value.strip():
        _fail(blank)
    return value


def validate_feature_payload(
    feature: InferenceFeature,
    *,
    image_base64: str | None = None,
    user_speech: str | None = None,
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
) -> None:
    """check request data for one feature"""
    if FEATURE_INPUT_TYPES[feature] is InputKind.IMAGE:
        raw_image = _require_non_blank(
            image_base64, Problem.IMAGE_REQUIRED, Problem.IMAGE_BLANK
        )
        if user_speech is not None:
            _fail(Problem.SPEECH_NOT_ALLOWED)
        validate_image_base64(raw_image, max_image_bytes)
        return

    if image_base64 is not None:
        _fail(Problem.IMAGE_NOT_ALLOWED)
    _require_non_blank(user_speech, Problem.SPEECH_REQUIRED, Problem.SPEECH_BLANK)
