from .classification import classification_metrics
from .detection import detection_metrics
from .keyword import keyword_metrics
from .ocr import ocr_errors
from .scene import scene_metrics
from .speech import speech_errors

SPEECH_RECOGNITION = "speech_recognition"
OCR = "ocr"
IMAGE_CLASSIFICATION = "image_classification"
INTENT_CLASSIFICATION = "intent_classification"
OBJECT_DETECTION = "object_detection"
KEYWORD_SPOTTING = "keyword_spotting"
SCENE_DESCRIPTION = "scene_description"

METRICS = {
    SPEECH_RECOGNITION: speech_errors,
    OCR: ocr_errors,
    IMAGE_CLASSIFICATION: classification_metrics,
    INTENT_CLASSIFICATION: classification_metrics,
    OBJECT_DETECTION: detection_metrics,
    KEYWORD_SPOTTING: keyword_metrics,
    SCENE_DESCRIPTION: scene_metrics,
}
