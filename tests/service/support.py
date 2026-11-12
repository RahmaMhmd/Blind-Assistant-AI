from src.contracts import InferenceFeature, OcrResult

OCR_PATH = f"/v1/infer/{InferenceFeature.OCR.value}"


class StubHandler:
    feature = InferenceFeature.OCR
    model_name = "stub"

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
