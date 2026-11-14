from enum import Enum


class InferenceFeature(str, Enum):
    OCR = "ocr"
    VOICE_COMMAND = "voice_command"
    SCENE_DESCRIPTION = "scene_description"
    CURRENCY_RECOGNITION = "currency_recognition"
    COLOR_DETECTION = "color_detection"
    TRAFFIC_SIGN_RECOGNITION = "traffic_sign_recognition"
    FACE_EXPRESSION = "face_expression"
    OBSTACLE_DETECTION = "obstacle_detection"


class SupportedLanguage(str, Enum):
    EGYPTIAN_ARABIC = "ar-EG"
    ARABIC = "ar"
    ENGLISH = "en"


class InputKind(str, Enum):
    IMAGE = "image"
    TEXT = "text"


TEXT_FEATURES = frozenset({InferenceFeature.VOICE_COMMAND})

FEATURE_INPUT_TYPES: dict[InferenceFeature, InputKind] = {
    feature: InputKind.TEXT if feature in TEXT_FEATURES else InputKind.IMAGE
    for feature in InferenceFeature
}
