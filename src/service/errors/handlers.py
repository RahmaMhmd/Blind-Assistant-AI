import logging
import traceback
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.contracts import ContractValidationError, ErrorCode, error_http_status

from ..runtime.context import ensure_context
from .builder import error_response
from .exceptions import ModelServiceError

logger = logging.getLogger(__name__)

MAPPED_HTTP_ERRORS = (
    ErrorCode.NOT_FOUND,
    ErrorCode.FILE_TOO_LARGE,
    ErrorCode.TOO_MANY_REQUESTS,
)

HTTP_STATUS_CODES = {
    error_http_status(code): code for code in MAPPED_HTTP_ERRORS
}


def _fallback_code(status: int) -> ErrorCode:
    if status >= HTTPStatus.INTERNAL_SERVER_ERROR:
        return ErrorCode.INTERNAL_SERVER_ERROR

    return ErrorCode.BAD_REQUEST


async def on_service_error(request: Request, exc: ModelServiceError):
    return error_response(
        request, exc.error_code, exc.details, exc.model_name
    )


async def on_contract_error(request: Request, exc: ContractValidationError):
    return error_response(request, exc.error_code, [exc.message])


async def on_invalid_request(request: Request, exc: RequestValidationError):
    details = [
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
        for item in exc.errors()
    ]

    return error_response(request, ErrorCode.VALIDATION_FAILED, details)


async def on_http_error(request: Request, exc: StarletteHTTPException):
    code = HTTP_STATUS_CODES.get(
        exc.status_code, _fallback_code(exc.status_code)
    )

    return error_response(request, code, status_code=exc.status_code)


async def on_unexpected_error(request: Request, exc: Exception):
    logger.error(
        "unhandled %s request=%s frames=%s",
        type(exc).__name__,
        ensure_context(request),
        "".join(traceback.format_tb(exc.__traceback__)),
    )

    return error_response(request, ErrorCode.INTERNAL_SERVER_ERROR)


EXCEPTION_HANDLERS = {
    ModelServiceError: on_service_error,
    ContractValidationError: on_contract_error,
    RequestValidationError: on_invalid_request,
    StarletteHTTPException: on_http_error,
    Exception: on_unexpected_error,
}


def register_error_handlers(app: FastAPI) -> None:
    """attach error handlers to app"""
    for exc_type, handler in EXCEPTION_HANDLERS.items():
        app.add_exception_handler(exc_type, handler)
