from pydantic import BaseModel

from src.contracts import InferenceFeature, OcrResult

from ..schemas import InferenceRequest

FAKE_OCR_RESULT = {
    "detectedLanguage": "ar",
    "extractedText": "نص تجريبي",
    "arabicSimplifiedText": "نص تجريبي",
    "confidence": 0.99,
}


class FakeOcrHandler:
    feature = InferenceFeature.OCR
    model_name = "fake-ocr"

    async def predict(self, payload: InferenceRequest) -> BaseModel:
        return OcrResult(**FAKE_OCR_RESULT)
