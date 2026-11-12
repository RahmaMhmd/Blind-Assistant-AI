import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.contracts import ContractValidationError, ErrorCode

from .builder import error_response
from .exceptions import ModelServiceError

logger = logging.getLogger(__name__)

HTTP_STATUS_CODES = {
    404: ErrorCode.NOT_FOUND,
    413: ErrorCode.FILE_TOO_LARGE,
    429: ErrorCode.TOO_MANY_REQUESTS,
}


def _fallback_code(status: int) -> ErrorCode:
    if status >= 500:
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
    logger.exception("unhandled error")

    return error_response(request, ErrorCode.INTERNAL_SERVER_ERROR)


EXCEPTION_HANDLERS = {
    ModelServiceError: on_service_error,
    ContractValidationError: on_contract_error,
    RequestValidationError: on_invalid_request,
    StarletteHTTPException: on_http_error,
    Exception: on_unexpected_error,
}


def register_error_handlers(app: FastAPI) -> None:
    for exc_type, handler in EXCEPTION_HANDLERS.items():
        app.add_exception_handler(exc_type, handler)
