import pytest
from pydantic import ValidationError

from src.contracts import (
    ContractValidationError,
    ErrorCode,
    ImageInput,
    InferenceFeature,
    VoiceCommandInput,
    validate_feature_payload,
    validate_image_base64,
)
from tests.helpers import JPEG_BYTES, JPEG_MIME, PNG_BYTES, PNG_MIME, data_uri


def test_image_input_matches_backend_field_name():
    payload = ImageInput(imageBase64=data_uri(JPEG_MIME, JPEG_BYTES))

    assert payload.imageBase64.startswith(f"data:{JPEG_MIME};base64,")


def test_voice_input_uses_user_speech():
    assert VoiceCommandInput(userSpeech="read this").userSpeech == "read this"


def test_voice_input_rejects_blank_text():
    with pytest.raises(ValidationError):
        VoiceCommandInput(userSpeech="   ")


def test_image_feature_requires_image():
    with pytest.raises(ContractValidationError) as exc:
        validate_feature_payload(InferenceFeature.OCR)

    assert exc.value.error_code == ErrorCode.VALIDATION_FAILED


def test_image_feature_rejects_blank_image():
    with pytest.raises(ContractValidationError) as exc:
        validate_feature_payload(InferenceFeature.OCR, image_base64="   ")

    assert exc.value.error_code == ErrorCode.VALIDATION_FAILED


def test_voice_feature_rejects_image():
    with pytest.raises(ContractValidationError):
        validate_feature_payload(
            InferenceFeature.VOICE_COMMAND,
            image_base64=data_uri(PNG_MIME, PNG_BYTES),
        )


@pytest.mark.parametrize(
    ("mime", "content"),
    [(JPEG_MIME, JPEG_BYTES), (PNG_MIME, PNG_BYTES)],
)
def test_valid_images_pass(mime, content):
    assert validate_image_base64(data_uri(mime, content)) == (mime, len(content))


def test_unsupported_mime_type_is_rejected():
    with pytest.raises(ContractValidationError) as exc:
        validate_image_base64(data_uri("image/webp", b"data"))

    assert exc.value.error_code == ErrorCode.UNSUPPORTED_IMAGE_FORMAT


def test_wrong_signature_is_rejected():
    with pytest.raises(ContractValidationError) as exc:
        validate_image_base64(data_uri(PNG_MIME, JPEG_BYTES))

    assert exc.value.error_code == ErrorCode.UNSUPPORTED_IMAGE_FORMAT


def test_invalid_base64_is_rejected():
    with pytest.raises(ContractValidationError) as exc:
        validate_image_base64(f"data:{JPEG_MIME};base64,%%%not-base64%%")

    assert exc.value.error_code == ErrorCode.VALIDATION_FAILED


def test_size_limit_is_a_parameter():
    with pytest.raises(ContractValidationError) as exc:
        validate_image_base64(data_uri(JPEG_MIME, JPEG_BYTES), max_bytes=4)

    assert exc.value.error_code == ErrorCode.FILE_TOO_LARGE
