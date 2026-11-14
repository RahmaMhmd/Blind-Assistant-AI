from typing import Annotated

from pydantic import BaseModel, Field

from .enums import FacialExpression, HapticPattern, MobileAction, VoiceIntent
from .features import InferenceFeature
from .models import MAX_CONFIDENCE, MIN_CONFIDENCE, MIN_DISTANCE

Confidence = Annotated[float, Field(ge=MIN_CONFIDENCE, le=MAX_CONFIDENCE)]
Distance = Annotated[float, Field(ge=MIN_DISTANCE)]

MIN_CURRENCY_DENOMINATION = 5
MAX_CURRENCY_DENOMINATION = 200
DEFAULT_CURRENCY = "EGP"


class FeatureResult(BaseModel):
    confidence: Confidence


class OcrResult(FeatureResult):
    detectedLanguage: str
    extractedText: str
    arabicSimplifiedText: str


class VoiceCommandResult(FeatureResult):
    userInput: str
    detectedIntent: VoiceIntent
    spokenResponse: str
    actionToTriggerOnMobile: MobileAction


class SceneDescriptionResult(FeatureResult):
    description: str
    detectedObjects: list[str]


class CurrencyResult(FeatureResult):
    denomination: int = Field(
        ge=MIN_CURRENCY_DENOMINATION,
        le=MAX_CURRENCY_DENOMINATION,
    )
    currency: str = DEFAULT_CURRENCY
    spokenAnnouncement: str
    condition: str


class ColorResult(FeatureResult):
    colorNameArabic: str
    colorNameEnglish: str
    hexCode: str
    disclaimer: str


class TrafficSignResult(FeatureResult):
    signName: str
    spokenMeaning: str


class FaceExpressionResult(FeatureResult):
    expression: FacialExpression
    spokenDescription: str


class ObstacleResult(FeatureResult):
    hazardDetected: bool
    obstacleType: str | None = None
    distanceMeters: Distance | None = None
    hapticAlertPattern: HapticPattern
    spokenWarning: str


RESULT_MODELS: dict[InferenceFeature, type[FeatureResult]] = {
    InferenceFeature.OCR: OcrResult,
    InferenceFeature.VOICE_COMMAND: VoiceCommandResult,
    InferenceFeature.SCENE_DESCRIPTION: SceneDescriptionResult,
    InferenceFeature.CURRENCY_RECOGNITION: CurrencyResult,
    InferenceFeature.COLOR_DETECTION: ColorResult,
    InferenceFeature.TRAFFIC_SIGN_RECOGNITION: TrafficSignResult,
    InferenceFeature.FACE_EXPRESSION: FaceExpressionResult,
    InferenceFeature.OBSTACLE_DETECTION: ObstacleResult,
}
