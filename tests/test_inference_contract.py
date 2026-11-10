import base64

import pytest

from pydantic import ValidationError

from src.contracts.errors import ErrorCode

from src.contracts.inference import (
    ErrorResponse,
    ImageInput,
    InferenceFeature,
    SuccessResponse,
    VoiceCommandInput,
)

from src.contracts.validators import (
    ContractValidationError,
    error_http_status,
    validate_feature_payload,
    validate_image_base64,
)


# =========================================================
# Helper
# =========================================================

def data_uri(mime: str, content: bytes) -> str:

    encoded = base64.b64encode(content).decode("ascii")

    return f"data:{mime};base64,{encoded}"


# =========================================================
# Input Tests
# =========================================================

def test_image_input_matches_backend_field_name():

    payload = ImageInput(
        imageBase64=data_uri(
            "image/jpeg",
            b"\xff\xd8\xffdemo",
        )
    )

    assert payload.imageBase64.startswith(
        "data:image/jpeg;base64,"
    )


def test_voice_input_uses_user_speech():

    payload = VoiceCommandInput(
        userSpeech="اقرأ الورقة دي"
    )

    assert payload.userSpeech == "اقرأ الورقة دي"


def test_ocr_requires_image():

    with pytest.raises(ContractValidationError):

        validate_feature_payload(
            InferenceFeature.OCR
        )


def test_ocr_accepts_valid_jpeg_data_uri():

    image = data_uri(
        "image/jpeg",
        b"\xff\xd8\xffsample",
    )

    validate_feature_payload(
        InferenceFeature.OCR,
        image_base64=image,
    )


def test_image_rejects_unsupported_mime_type():

    image = data_uri(
        "image/webp",
        b"webp-data",
    )

    with pytest.raises(ContractValidationError) as exc:

        validate_image_base64(image)

    assert (
        exc.value.error_code
        == ErrorCode.UNSUPPORTED_IMAGE_FORMAT
    )


def test_image_rejects_invalid_base64():

    with pytest.raises(ContractValidationError):

        validate_image_base64(
            "data:image/jpeg;base64,%%%not-base64%%"
        )


def test_voice_command_requires_non_empty_text():

    with pytest.raises(ValidationError):

        VoiceCommandInput(
            userSpeech="   "
        )


# =========================================================
# Success Response Tests
# =========================================================

def test_success_response_contains_ai_tracking_fields():

    response = SuccessResponse(

        requestId="req-12345",

        modelName="PaddleOCR",

        processingTimeMs=245.6,

        data={
            "extractedText": "أهلاً"
        },
    )

    dumped = response.model_dump()

    assert dumped["success"] is True

    assert dumped["statusCode"] == 200

    assert "timestamp" in dumped

    assert dumped["requestId"] == "req-12345"

    assert dumped["modelName"] == "PaddleOCR"

    assert dumped["processingTimeMs"] == 245.6

    assert dumped["data"]["extractedText"] == "أهلاً"


def test_success_response_rejects_negative_processing_time():

    with pytest.raises(ValidationError):

        SuccessResponse(

            requestId="req-12345",

            modelName="PaddleOCR",

            processingTimeMs=-1,

            data={},
        )


# =========================================================
# Error Response Tests
# =========================================================

def test_error_response_contains_arabic_message():

    response = ErrorResponse.from_code(

        status_code=422,

        error_code=ErrorCode.IMAGE_BLURRY,

        request_id="req-12345",

        model_name="PaddleOCR",

        processing_time_ms=120.5,
    )

    dumped = response.model_dump()

    assert dumped["success"] is False

    assert dumped["statusCode"] == 422

    assert dumped["errorCode"] == "ERR_IMAGE_BLURRY"

    assert (
        dumped["message"]
        == "الصورة مشوشة أو غير واضحة، يرجى إعادة التقاطها بثبات"
    )

    assert dumped["requestId"] == "req-12345"

    assert dumped["modelName"] == "PaddleOCR"

    assert dumped["processingTimeMs"] == 120.5

    assert "timestamp" in dumped


def test_error_response_allows_missing_model_information():

    response = ErrorResponse.from_code(

        status_code=400,

        error_code=ErrorCode.VALIDATION_FAILED,

        request_id="req-67890",
    )

    assert response.modelName is None

    assert response.processingTimeMs is None

    assert response.requestId == "req-67890"


def test_error_response_requires_request_id():

    with pytest.raises(ValidationError):

        ErrorResponse(

            statusCode=500,

            errorCode=ErrorCode.INTERNAL_SERVER_ERROR,

            message="حدث خطأ غير متوقع في الخادم",
        )


# =========================================================
# HTTP Status Tests
# =========================================================

def test_file_too_large_status_matches_backend():

    assert (
        error_http_status(ErrorCode.FILE_TOO_LARGE)
        == 413
    )