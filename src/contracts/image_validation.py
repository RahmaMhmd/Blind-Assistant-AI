import base64
import binascii

from .errors import ErrorCode
from .image_rules import DATA_URI_PATTERN, DEFAULT_MAX_IMAGE_BYTES, IMAGE_SIGNATURES
from .validation_error import ContractValidationError
from .validation_messages import Problem


def _decode(encoded: str) -> bytes:
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise ContractValidationError(
            ErrorCode.VALIDATION_FAILED, Problem.IMAGE_DATA
        ) from None

    if not decoded:
        raise ContractValidationError(ErrorCode.VALIDATION_FAILED, Problem.IMAGE_EMPTY)

    return decoded


def validate_image_base64(
    image_base64: str,
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
) -> tuple[str, int]:
    """check image data format and byte size"""
    match = DATA_URI_PATTERN.match(image_base64.strip())

    if not match:
        raise ContractValidationError(
            ErrorCode.UNSUPPORTED_IMAGE_FORMAT, Problem.IMAGE_FORMAT
        )

    mime_type, encoded = match.groups()
    decoded = _decode(encoded)

    if len(decoded) > max_bytes:
        raise ContractValidationError(ErrorCode.FILE_TOO_LARGE, Problem.IMAGE_SIZE)

    if not decoded.startswith(IMAGE_SIGNATURES[mime_type]):
        raise ContractValidationError(
            ErrorCode.UNSUPPORTED_IMAGE_FORMAT, Problem.IMAGE_CONTENT
        )

    return mime_type, len(decoded)
