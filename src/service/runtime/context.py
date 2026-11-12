import re
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

REQUEST_ID_HEADER = "X-Request-ID"

_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def ensure_context(request: Request) -> str:
    if not hasattr(request.state, "request_id"):
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        valid = _SAFE_REQUEST_ID.match(incoming)

        request.state.request_id = incoming if valid else str(uuid.uuid4())
        request.state.started_at = time.perf_counter()

    return request.state.request_id


def elapsed_ms(request: Request) -> float:
    ensure_context(request)

    return (time.perf_counter() - request.state.started_at) * 1000


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = ensure_context(request)
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id

        return response
