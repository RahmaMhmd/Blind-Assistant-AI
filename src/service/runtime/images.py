import base64
import binascii

from src.contracts import ErrorCode
from src.contracts.image_rules import DATA_URI_PATTERN

from ..errors.exceptions import ModelServiceError


def decode_data_uri(value: str) -> bytes:
    """pipeline validation runs first so a failure here is a service bug"""
    match = DATA_URI_PATTERN.match(value.strip())

    if not match:
        raise ModelServiceError(ErrorCode.UNSUPPORTED_IMAGE_FORMAT)

    try:
        return base64.b64decode(match.group(2), validate=True)
    except binascii.Error:
        raise ModelServiceError(ErrorCode.VALIDATION_FAILED) from None
