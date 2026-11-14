from fastapi import Request
from fastapi.responses import JSONResponse

from src.contracts import ErrorCode, ErrorResponse, error_http_status

from ..runtime.context import REQUEST_ID_HEADER, elapsed_ms, ensure_context


def error_response(
    request: Request,
    error_code: ErrorCode,
    details: list[str] | None = None,
    model_name: str | None = None,
    status_code: int | None = None,
) -> JSONResponse:
    """make standard json error response"""
    status = status_code or error_http_status(error_code)
    request_id = ensure_context(request)

    body = ErrorResponse.from_code(
        status_code=status,
        error_code=error_code,
        request_id=request_id,
        details=details,
        model_name=model_name,
        processing_time_ms=elapsed_ms(request),
    )

    return JSONResponse(
        status_code=status,
        content=body.model_dump(mode="json"),
        headers={REQUEST_ID_HEADER: request_id},
    )
