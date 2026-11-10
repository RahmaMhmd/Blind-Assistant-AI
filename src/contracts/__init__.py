"""Unified Baseer AI contracts."""

from .errors import (
    ErrorCode,
    ERROR_MESSAGES_AR,
    get_arabic_error_message,
)

from .inference import (
    InferenceFeature,
    SupportedLanguage,

    ImageInput,
    VoiceCommandInput,

    SuccessResponse,
    ErrorResponse,

    OcrResult,
    VoiceCommandResult,
    SceneDescriptionResult,
    CurrencyResult,
    ColorResult,
    TrafficSignResult,
    FaceExpressionResult,
    ObstacleResult,
)

from .validators import (
    ContractValidationError,
    validate_feature_payload,
    validate_image_base64,
    error_http_status,
)


__all__ = [

    "ErrorCode",
    "ERROR_MESSAGES_AR",
    "get_arabic_error_message",

    "InferenceFeature",
    "SupportedLanguage",

    "ImageInput",
    "VoiceCommandInput",

    "SuccessResponse",
    "ErrorResponse",

    "OcrResult",
    "VoiceCommandResult",
    "SceneDescriptionResult",
    "CurrencyResult",
    "ColorResult",
    "TrafficSignResult",
    "FaceExpressionResult",
    "ObstacleResult",

    "ContractValidationError",
    "validate_feature_payload",
    "validate_image_base64",
    "error_http_status",
]