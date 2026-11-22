from .classification import classification_metrics
from .ocr import ocr_errors
from .speech import speech_errors

SPEECH_RECOGNITION = "speech_recognition"
OCR = "ocr"
IMAGE_CLASSIFICATION = "image_classification"
INTENT_CLASSIFICATION = "intent_classification"

# one metric per task type, so the same task is always measured the same way
METRICS = {
    SPEECH_RECOGNITION: speech_errors,
    OCR: ocr_errors,
    IMAGE_CLASSIFICATION: classification_metrics,
    INTENT_CLASSIFICATION: classification_metrics,
}
