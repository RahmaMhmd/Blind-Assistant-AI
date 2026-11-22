import asyncio
from dataclasses import dataclass
from typing import Protocol

from src.contracts import ErrorCode, InferenceFeature, OcrResult

from ..errors.exceptions import ModelServiceError
from ..runtime.images import decode_data_uri
from ..schemas import InferenceRequest
from .ocr_text import detect_language

LINE_SEPARATOR = "\n"


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float


class OcrEngine(Protocol):
    def read(self, image: bytes) -> list[OcrLine]: ...


class OcrHandler:
    """arabicSimplifiedText stays equal to extractedText until backend defines it"""

    feature = InferenceFeature.OCR

    def __init__(self, engine: OcrEngine, model_name: str, model_version: str):
        self._engine = engine
        self.model_name = model_name
        self.model_version = model_version

    async def predict(self, payload: InferenceRequest) -> OcrResult:
        image = decode_data_uri(payload.imageBase64)
        lines = await asyncio.to_thread(self._engine.read, image)
        lines = [line for line in lines if line.text.strip()]

        if not lines:
            raise ModelServiceError(
                ErrorCode.NO_TEXT_DETECTED, model_name=self.model_name
            )

        text = LINE_SEPARATOR.join(line.text.strip() for line in lines)

        return OcrResult(
            confidence=sum(line.confidence for line in lines) / len(lines),
            detectedLanguage=detect_language(text),
            extractedText=text,
            arabicSimplifiedText=text,
        )
