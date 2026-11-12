import base64
import binascii
import re

from .errors import ErrorCode

from .inference import (
    FEATURE_INPUT_TYPES,
    InferenceFeature,
)


# =========================================================
# 1. Image Configuration
# =========================================================

SUPPORTED_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
}

MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024

_DATA_URI_PATTERN = re.compile(
    r"^data:(image/(?:jpeg|png));base64,(.+)$",
    re.DOTALL,
)


# =========================================================
# 2. Validation Exception
# =========================================================

class ContractValidationError(ValueError):

    def __init__(
        self,
        error_code: ErrorCode,
        message: str | None = None,
    ):

        self.error_code = error_code

        self.message = message or "بيانات الطلب غير صالحة"

        super().__init__(self.message)


# =========================================================
# 3. Feature Payload Validation
# =========================================================

def validate_feature_payload(
    feature: InferenceFeature,
    *,
    image_base64: str | None = None,
    user_speech: str | None = None,
) -> None:

    expected_input = FEATURE_INPUT_TYPES[feature]

    # -------------------------------------
    # Image-based features
    # -------------------------------------

    if expected_input == "image":

        if not image_base64:

            raise ContractValidationError(
                ErrorCode.VALIDATION_FAILED,
                "يرجى إرفاق صورة لإتمام الطلب",
            )

        if user_speech is not None:

            raise ContractValidationError(
                ErrorCode.VALIDATION_FAILED,
                "الميزة دي بتستقبل صورة، مش نص أمر صوتي",
            )

        validate_image_base64(image_base64)

        return

    # -------------------------------------
    # Text-based features
    # -------------------------------------

    if expected_input == "text":

        if image_base64 is not None:

            raise ContractValidationError(
                ErrorCode.VALIDATION_FAILED,
                "الميزة دي بتستقبل نص الأمر الصوتي، مش صورة",
            )

        if user_speech is None or not user_speech.strip():

            raise ContractValidationError(
                ErrorCode.VALIDATION_FAILED,
                "من فضلك قل الأمر الصوتي أو أرسل نصه",
            )

        return

    raise ContractValidationError(
        ErrorCode.VALIDATION_FAILED,
        "نوع البيانات المطلوبة للميزة دي غير محدد",
    )


# =========================================================
# 4. Base64 Image Validation
# =========================================================

def validate_image_base64(
    image_base64: str,
) -> tuple[str, int]:

    match = _DATA_URI_PATTERN.match(
        image_base64.strip()
    )

    if not match:

        raise ContractValidationError(
            ErrorCode.UNSUPPORTED_IMAGE_FORMAT,
            "صيغة الصورة غير مدعومة، يرجى إرسال صورة JPEG أو PNG",
        )

    mime_type, encoded_content = match.groups()

    # -------------------------------------
    # Decode Base64
    # -------------------------------------

    try:

        decoded = base64.b64decode(
            encoded_content,
            validate=True,
        )

    except (binascii.Error, ValueError):

        raise ContractValidationError(
            ErrorCode.VALIDATION_FAILED,
            "بيانات الصورة غير صالحة، يرجى إعادة إرسال الصورة",
        ) from None

    # -------------------------------------
    # Check empty image
    # -------------------------------------

    if not decoded:

        raise ContractValidationError(
            ErrorCode.VALIDATION_FAILED,
            "الصورة المرسلة فارغة",
        )

    # -------------------------------------
    # Check image size
    # -------------------------------------

    if len(decoded) > MAX_IMAGE_SIZE_BYTES:

        raise ContractValidationError(
            ErrorCode.FILE_TOO_LARGE,
            "حجم الصورة كبير جداً وتجاوز الحد الأقصى",
        )

    # -------------------------------------
    # Check JPEG signature
    # -------------------------------------

    if mime_type == "image/jpeg":

        if not decoded.startswith(b"\xff\xd8\xff"):

            raise ContractValidationError(
                ErrorCode.UNSUPPORTED_IMAGE_FORMAT,
                "محتوى الملف مش صورة JPEG سليمة",
            )

    # -------------------------------------
    # Check PNG signature
    # -------------------------------------

    if mime_type == "image/png":

        if not decoded.startswith(b"\x89PNG\r\n\x1a\n"):

            raise ContractValidationError(
                ErrorCode.UNSUPPORTED_IMAGE_FORMAT,
                "محتوى الملف مش صورة PNG سليمة",
            )

    return mime_type, len(decoded)


# =========================================================
# 5. HTTP Status Mapping
# =========================================================

def error_http_status(
    error_code: ErrorCode,
) -> int:


    mapping = {

        ErrorCode.VALIDATION_FAILED: 400,

        ErrorCode.BAD_REQUEST: 400,

        ErrorCode.INVALID_QR_CODE: 400,

        ErrorCode.UNAUTHORIZED: 401,

        ErrorCode.FORBIDDEN: 403,

        ErrorCode.NOT_FOUND: 404,

        ErrorCode.QR_NOT_FOUND: 404,

        ErrorCode.DESTINATION_NOT_FOUND: 404,

        ErrorCode.FILE_TOO_LARGE: 413,

        ErrorCode.IMAGE_BLURRY: 422,

        ErrorCode.IMAGE_TOO_DARK: 422,

        ErrorCode.NO_TEXT_DETECTED: 422,

        ErrorCode.UNSUPPORTED_IMAGE_FORMAT: 422,

        ErrorCode.INTENT_NOT_RECOGNIZED: 422,

        ErrorCode.ROUTE_CALCULATION_FAILED: 422,

        ErrorCode.LOW_CONFIDENCE: 422,

        ErrorCode.TOO_MANY_REQUESTS: 429,

        ErrorCode.INTERNAL_SERVER_ERROR: 500,

        ErrorCode.SERVICE_UNAVAILABLE: 503,
    }

    return mapping[error_code]