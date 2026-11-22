import asyncio

import pytest

from src.contracts import ErrorCode, InferenceFeature
from src.service.errors.exceptions import ModelServiceError
from src.service.handlers.ocr import OcrHandler, OcrLine
from src.service.handlers.ocr_text import detect_language
from src.service.runtime.images import decode_data_uri
from src.service.schemas import InferenceRequest
from tests.helpers import JPEG_BYTES, image_body

NAME = "fake-ocr"
VERSION = "0.0.1"


class FakeEngine:
    def __init__(self, lines):
        self.lines = lines
        self.received = None

    def read(self, image):
        self.received = image

        return self.lines


def make_handler(lines):
    engine = FakeEngine(lines)

    return OcrHandler(engine, NAME, VERSION), engine


def run(handler):
    return asyncio.run(handler.predict(InferenceRequest(**image_body())))


def test_returns_joined_text_and_mean_confidence():
    lines = [OcrLine("مرحبا", 0.9), OcrLine("بالعالم", 0.7)]
    handler, engine = make_handler(lines)
    result = run(handler)

    assert result.extractedText == "مرحبا\nبالعالم"
    assert result.arabicSimplifiedText == result.extractedText
    assert result.detectedLanguage == "ar"
    assert result.confidence == pytest.approx(0.8)
    assert engine.received == JPEG_BYTES


@pytest.mark.parametrize("lines", [[], [OcrLine("   ", 0.9)]])
def test_empty_text_raises_no_text_detected(lines):
    handler, _ = make_handler(lines)

    with pytest.raises(ModelServiceError) as caught:
        run(handler)

    assert caught.value.error_code is ErrorCode.NO_TEXT_DETECTED
    assert caught.value.model_name == NAME


def test_handler_follows_model_handler_contract():
    handler, _ = make_handler([])

    assert handler.feature is InferenceFeature.OCR
    assert (handler.model_name, handler.model_version) == (NAME, VERSION)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("نص عربي", "ar"),
        ("plain text", "en"),
        ("123", "en"),
        ("نص عربي long english text here", "en"),
    ],
)
def test_language_follows_letter_majority(text, expected):
    assert detect_language(text) == expected


def test_decode_data_uri_round_trips_bytes():
    assert decode_data_uri(image_body()["imageBase64"]) == JPEG_BYTES


@pytest.mark.parametrize("value", ["not a data uri", "data:image/jpeg;base64,@@@"])
def test_decode_data_uri_rejects_bad_input(value):
    with pytest.raises(ModelServiceError):
        decode_data_uri(value)
