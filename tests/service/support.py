from src.contracts import InferenceFeature, OcrResult
from src.service.routes.paths import infer_path

feature_path = infer_path
OCR_PATH = infer_path(InferenceFeature.OCR)


class StubHandler:
    feature = InferenceFeature.OCR
    model_name = "stub"
    model_version = "0.0.0-stub"

    def __init__(self, action):
        self.action = action

    async def predict(self, payload):
        return await self.action()


def stub_result(confidence: float) -> OcrResult:
    return OcrResult(
        detectedLanguage="en",
        extractedText="text",
        arabicSimplifiedText="text",
        confidence=confidence,
    )
