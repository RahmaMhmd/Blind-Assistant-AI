from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from .errors import ErrorCode, get_arabic_error_message


# =========================================================
# 1. AI Features
# =========================================================

class InferenceFeature(str, Enum):
    """
    AI-backed features supported by Baseer.
    """

    OCR = "ocr"

    VOICE_COMMAND = "voice_command"

    SCENE_DESCRIPTION = "scene_description"

    CURRENCY_RECOGNITION = "currency_recognition"

    COLOR_DETECTION = "color_detection"

    TRAFFIC_SIGN_RECOGNITION = "traffic_sign_recognition"

    FACE_EXPRESSION = "face_expression"

    OBSTACLE_DETECTION = "obstacle_detection"


# =========================================================
# 2. Supported Languages
# =========================================================

class SupportedLanguage(str, Enum):

    EGYPTIAN_ARABIC = "ar-EG"

    ARABIC = "ar"

    ENGLISH = "en"


# =========================================================
# 3. Input Schemas
# =========================================================

class ImageInput(BaseModel):

    model_config = ConfigDict(extra="forbid")

    imageBase64: str = Field(min_length=1)

    @field_validator("imageBase64")
    @classmethod
    def validate_image_base64_not_blank(cls, value: str) -> str:

        if not value.strip():

            raise ValueError(
                "الصورة المرسلة لا يمكن أن تكون فارغة"
            )

        return value


class VoiceCommandInput(BaseModel):

    model_config = ConfigDict(extra="forbid")

    userSpeech: str = Field(min_length=1)

    @field_validator("userSpeech")
    @classmethod
    def validate_user_speech(cls, value: str) -> str:

        if not value.strip():

            raise ValueError(
                "نص الأمر الصوتي لا يمكن أن يكون فارغاً"
            )

        return value


# =========================================================
# 4. Success Response
# =========================================================

class SuccessResponse(BaseModel):
  
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    success: bool = True

    statusCode: int = Field(
        default=200,
        ge=200,
        le=299,
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # AI Tracking Information

    requestId: str = Field(
        min_length=1
    )

    modelName: str = Field(
        min_length=1
    )

    processingTimeMs: float = Field(
        ge=0
    )

    # Feature-specific result

    data: dict[str, Any]


# =========================================================
# 5. Error Response
# =========================================================

class ErrorResponse(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    success: bool = False

    statusCode: int = Field(
        ge=400,
        le=599,
    )

    errorCode: ErrorCode

    message: str = Field(
        min_length=1
    )

    details: list[str] = Field(
        default_factory=list
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # AI Tracking Information

    requestId: str = Field(
        min_length=1
    )

    modelName: str | None = None

    processingTimeMs: float | None = Field(
        default=None,
        ge=0,
    )

    @classmethod
    def from_code(
        cls,
        status_code: int,
        error_code: ErrorCode,
        request_id: str,
        details: list[str] | None = None,
        model_name: str | None = None,
        processing_time_ms: float | None = None,
    ) -> "ErrorResponse":

        return cls(
            success=False,
            statusCode=status_code,
            errorCode=error_code,

            
            message=get_arabic_error_message(error_code),

            details=details or [],

            requestId=request_id,

            modelName=model_name,

            processingTimeMs=processing_time_ms,
        )


# =========================================================
# 6. Feature Input Mapping
# =========================================================

FEATURE_INPUT_TYPES: dict[InferenceFeature, str] = {

    InferenceFeature.OCR: "image",

    InferenceFeature.SCENE_DESCRIPTION: "image",

    InferenceFeature.CURRENCY_RECOGNITION: "image",

    InferenceFeature.COLOR_DETECTION: "image",

    InferenceFeature.TRAFFIC_SIGN_RECOGNITION: "image",

    InferenceFeature.FACE_EXPRESSION: "image",

    InferenceFeature.OBSTACLE_DETECTION: "image",

    InferenceFeature.VOICE_COMMAND: "text",
}


# =========================================================
# 7. Feature Result Schemas
# =========================================================

class OcrResult(BaseModel):

    detectedLanguage: str

    extractedText: str

    arabicSimplifiedText: str

    confidence: float = Field(
        ge=0,
        le=1,
    )


class VoiceCommandResult(BaseModel):

    userInput: str

    detectedIntent: str

    confidence: float = Field(
        ge=0,
        le=1,
    )

    spokenResponse: str

    actionToTriggerOnMobile: str


class SceneDescriptionResult(BaseModel):

    description: str

    detectedObjects: list[str]

    confidence: float = Field(
        ge=0,
        le=1,
    )


class CurrencyResult(BaseModel):

    denomination: int = Field(
        ge=5,
        le=200,
    )

    currency: str = "EGP"

    spokenAnnouncement: str

    condition: str

    confidence: float = Field(
        ge=0,
        le=1,
    )


class ColorResult(BaseModel):

    colorNameArabic: str

    colorNameEnglish: str

    hexCode: str

    disclaimer: str

    confidence: float = Field(
        ge=0,
        le=1,
    )


class TrafficSignResult(BaseModel):

    signName: str

    spokenMeaning: str

    confidence: float = Field(
        ge=0,
        le=1,
    )


class FaceExpressionResult(BaseModel):

    expression: str

    spokenDescription: str

    confidence: float = Field(
        ge=0,
        le=1,
    )


class ObstacleResult(BaseModel):

    hazardDetected: bool

    obstacleType: str | None = None

    distanceMeters: float | None = Field(
        default=None,
        ge=0,
    )

    hapticAlertPattern: str

    spokenWarning: str

    confidence: float = Field(
        ge=0,
        le=1,
    )