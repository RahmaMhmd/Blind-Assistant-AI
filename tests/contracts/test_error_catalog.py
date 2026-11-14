import re
from http import HTTPStatus
from pathlib import Path

from src.contracts import (
    ErrorCode,
    ai_emittable_codes,
    error_http_status,
    error_specs,
    get_error_message,
)
from src.contracts.catalog import ERROR_CODES_FILE, read_json
from src.contracts.models import ERROR_STATUS_MAX, ERROR_STATUS_MIN

SERVICE_DIR = Path(__file__).resolve().parents[2] / "src" / "service"
CODE_USAGE = re.compile(r"ErrorCode\.([A-Z_]+)")


def test_enum_matches_catalog_file():
    rows = read_json(ERROR_CODES_FILE)["codes"]

    assert {code.value for code in ErrorCode} == {row["code"] for row in rows}
    assert set(error_specs()) == set(ErrorCode)


def test_every_code_has_status_and_message():
    for code in ErrorCode:
        assert ERROR_STATUS_MIN <= error_http_status(code) <= ERROR_STATUS_MAX
        assert get_error_message(code)


def test_new_codes_have_agreed_status():
    assert error_http_status(ErrorCode.REQUEST_TIMEOUT) == HTTPStatus.GATEWAY_TIMEOUT
    assert error_http_status(ErrorCode.LOW_CONFIDENCE) == HTTPStatus.UNPROCESSABLE_ENTITY
    assert error_http_status(ErrorCode.FILE_TOO_LARGE) == HTTPStatus.REQUEST_ENTITY_TOO_LARGE


def test_ai_never_emits_backend_only_codes():
    backend_only = {
        ErrorCode.UNAUTHORIZED,
        ErrorCode.FORBIDDEN,
        ErrorCode.QR_NOT_FOUND,
        ErrorCode.INVALID_QR_CODE,
        ErrorCode.DESTINATION_NOT_FOUND,
        ErrorCode.ROUTE_CALCULATION_FAILED,
    }

    assert not backend_only & ai_emittable_codes()


def test_service_only_raises_emittable_codes():
    used = {
        name
        for path in SERVICE_DIR.rglob("*.py")
        for name in CODE_USAGE.findall(path.read_text(encoding="utf-8"))
    }

    assert {ErrorCode[name] for name in used} <= ai_emittable_codes()
