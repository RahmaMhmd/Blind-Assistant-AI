from enum import Enum


class ErrorCode(str, Enum):
    """Error codes currently declared by Baseer's Backend OpenAPI."""

    INTERNAL_SERVER_ERROR = "ERR_INTERNAL_SERVER_ERROR"
    VALIDATION_FAILED = "ERR_VALIDATION_FAILED"
    UNAUTHORIZED = "ERR_UNAUTHORIZED"
    FORBIDDEN = "ERR_FORBIDDEN"
    NOT_FOUND = "ERR_NOT_FOUND"
    BAD_REQUEST = "ERR_BAD_REQUEST"
    FILE_TOO_LARGE = "ERR_FILE_TOO_LARGE"
    TOO_MANY_REQUESTS = "ERR_TOO_MANY_REQUESTS"
    SERVICE_UNAVAILABLE = "ERR_SERVICE_UNAVAILABLE"

    IMAGE_BLURRY = "ERR_IMAGE_BLURRY"
    IMAGE_TOO_DARK = "ERR_IMAGE_TOO_DARK"
    NO_TEXT_DETECTED = "ERR_NO_TEXT_DETECTED"
    UNSUPPORTED_IMAGE_FORMAT = "ERR_UNSUPPORTED_IMAGE_FORMAT"

    INTENT_NOT_RECOGNIZED = "ERR_INTENT_NOT_RECOGNIZED"

    QR_NOT_FOUND = "ERR_QR_NOT_FOUND"
    INVALID_QR_CODE = "ERR_INVALID_QR_CODE"
    DESTINATION_NOT_FOUND = "ERR_DESTINATION_NOT_FOUND"
    ROUTE_CALCULATION_FAILED = "ERR_ROUTE_CALCULATION_FAILED"
    LOW_CONFIDENCE = "ERR_LOW_CONFIDENCE"


ERROR_MESSAGES_AR: dict[ErrorCode, str] = {

    ErrorCode.INTERNAL_SERVER_ERROR:
        "حدث خطأ غير متوقع في الخادم",

    ErrorCode.VALIDATION_FAILED:
        "بيانات الطلب غير صالحة",

    ErrorCode.UNAUTHORIZED:
        "يجب تسجيل الدخول أولاً للمتابعة",

    ErrorCode.FORBIDDEN:
        "ليس لديك صلاحية لتنفيذ هذا الإجراء",

    ErrorCode.NOT_FOUND:
        "المطلوب غير موجود",

    ErrorCode.BAD_REQUEST:
        "الطلب غير صالح، يرجى مراجعة البيانات وإعادة المحاولة",

    ErrorCode.FILE_TOO_LARGE:
        "حجم الصورة كبير جداً وتجاوز الحد الأقصى",

    ErrorCode.TOO_MANY_REQUESTS:
        "تم إرسال طلبات كثيرة في وقت قصير، يرجى الانتظار والمحاولة مرة أخرى",

    ErrorCode.SERVICE_UNAVAILABLE:
        "الخدمة غير متاحة حالياً، يرجى المحاولة مرة أخرى لاحقاً",

    ErrorCode.IMAGE_BLURRY:
        "الصورة مشوشة أو غير واضحة، يرجى إعادة التقاطها بثبات",

    ErrorCode.IMAGE_TOO_DARK:
        "الإضاءة ضعيفة جداً، يرجى تشغيل الفلاش أو الانتقال لمكان مضيء",

    ErrorCode.NO_TEXT_DETECTED:
        "لم يتم العثور على أي نص في الصورة الملتقطة",

    ErrorCode.UNSUPPORTED_IMAGE_FORMAT:
        "صيغة الصورة غير مدعومة، يرجى إرسال صورة JPEG أو PNG",

    ErrorCode.INTENT_NOT_RECOGNIZED:
        "عذراً، لم أفهم طلبك، يرجى المحاولة بصيغة أخرى",

    ErrorCode.QR_NOT_FOUND:
        "لم يتم العثور على أي كود QR في الإطار",

    ErrorCode.INVALID_QR_CODE:
        "كود الـ QR غير صالح أو تالف",

    ErrorCode.DESTINATION_NOT_FOUND:
        "المكان أو الوجهة المطلوبة غير مسجلة بالنظام",

    ErrorCode.ROUTE_CALCULATION_FAILED:
        "تعذر حساب المسار إلى الوجهة المطلوبة",

    ErrorCode.LOW_CONFIDENCE:
        "لم أتمكن من التأكد من النتيجة، يرجى إعادة المحاولة بشكل أوضح",
}


def get_arabic_error_message(code: ErrorCode) -> str:
    """
    Return the Arabic message corresponding to an error code.
    """

    return ERROR_MESSAGES_AR[code]