import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.contracts.patterns import REQUEST_ID_PATTERN

REQUEST_ID_HEADER = "X-Request-ID"
MS_PER_SECOND = 1000


def ensure_context(request: Request) -> str:
    """set request id and start time"""
    if not hasattr(request.state, "request_id"):
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        valid = REQUEST_ID_PATTERN.match(incoming)

        request.state.request_id = incoming if valid else str(uuid.uuid4())
        request.state.started_at = time.perf_counter()

    return request.state.request_id


def elapsed_ms(request: Request) -> float:
    ensure_context(request)

    return (time.perf_counter() - request.state.started_at) * MS_PER_SECOND


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = ensure_context(request)
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id

        return response
