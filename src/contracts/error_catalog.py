from dataclasses import dataclass
from functools import lru_cache

from .catalog import ERROR_CODES_FILE, read_json
from .errors import ErrorCode

AI_OWNER = "ai"


@dataclass(frozen=True)
class ErrorSpec:
    code: ErrorCode
    http: int
    owners: frozenset[str]
    retry: str
    message_key: str
    message: str


@lru_cache
def error_specs() -> dict[ErrorCode, ErrorSpec]:
    """load error rules from contract json"""
    specs = {}

    for row in read_json(ERROR_CODES_FILE)["codes"]:
        code = ErrorCode(row["code"])
        specs[code] = ErrorSpec(
            code=code,
            http=row["http"],
            owners=frozenset(row["owner"]),
            retry=row["retry"],
            message_key=row["messageKey"],
            message=row["fallbackMessage"],
        )

    return specs


def error_http_status(code: ErrorCode) -> int:
    return error_specs()[code].http


def get_error_message(code: ErrorCode) -> str:
    return error_specs()[code].message


def ai_emittable_codes() -> frozenset[ErrorCode]:
    return frozenset(
        code for code, spec in error_specs().items() if AI_OWNER in spec.owners
    )
