from http import HTTPStatus

import pytest
from pydantic import ValidationError

from src.contracts import (
    ErrorCode,
    ErrorResponse,
    InferenceFeature,
    SuccessResponse,
    error_http_status,
    get_error_message,
)
from src.service.settings import ServiceSettings

REQUEST_ID = "req-12345"
SETTINGS = ServiceSettings()
OCR_MODEL = f"{InferenceFeature.OCR.value}{SETTINGS.fake_model_suffix}"
OCR_VERSION = SETTINGS.fake_model_version


def success(**overrides) -> SuccessResponse:
    fields = {
        "requestId": REQUEST_ID,
        "modelName": OCR_MODEL,
        "modelVersion": OCR_VERSION,
        "processingTimeMs": 245.6,
        "data": {"extractedText": "hello"},
    }

    return SuccessResponse(**{**fields, **overrides})


def test_success_response_carries_tracking_fields():
    dumped = success().model_dump()

    assert dumped["success"] is True
    assert dumped["statusCode"] == HTTPStatus.OK
    assert dumped["requestId"] == REQUEST_ID
    assert dumped["modelVersion"] == OCR_VERSION
    assert "timestamp" in dumped


def test_success_response_rejects_negative_time():
    with pytest.raises(ValidationError):
        success(processingTimeMs=-1)


def test_success_response_requires_model_version():
    with pytest.raises(ValidationError):
        SuccessResponse(
            requestId=REQUEST_ID,
            modelName=OCR_MODEL,
            processingTimeMs=1,
            data={},
        )


def test_error_response_uses_catalog_message():
    code = ErrorCode.IMAGE_BLURRY
    dumped = ErrorResponse.from_code(
        status_code=error_http_status(code),
        error_code=code,
        request_id=REQUEST_ID,
        model_name=OCR_MODEL,
        processing_time_ms=120.5,
    ).model_dump()

    assert dumped["success"] is False
    assert dumped["errorCode"] == code.value
    assert dumped["message"] == get_error_message(code)
    assert dumped["details"] == []


def test_error_response_requires_request_id():
    with pytest.raises(ValidationError):
        ErrorResponse(
            statusCode=HTTPStatus.INTERNAL_SERVER_ERROR,
            errorCode=ErrorCode.INTERNAL_SERVER_ERROR,
            message="failed",
        )
