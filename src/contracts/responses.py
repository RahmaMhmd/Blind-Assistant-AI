from datetime import datetime, timezone
from http import HTTPStatus
from typing import Annotated, Any

from pydantic import Field

from .error_catalog import get_error_message
from .errors import ErrorCode
from .models import (
    ERROR_STATUS_MAX,
    ERROR_STATUS_MIN,
    SUCCESS_STATUS_MAX,
    SUCCESS_STATUS_MIN,
    ContractResponseModel,
)

NonEmpty = Annotated[str, Field(min_length=1)]
Millis = Annotated[float, Field(ge=0)]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class BaseEnvelope(ContractResponseModel):
    timestamp: datetime = Field(default_factory=_now)
    requestId: NonEmpty


class SuccessResponse(BaseEnvelope):
    """standard success response body"""

    success: bool = True
    statusCode: int = Field(
        default=HTTPStatus.OK,
        ge=SUCCESS_STATUS_MIN,
        le=SUCCESS_STATUS_MAX,
    )
    modelName: NonEmpty
    modelVersion: NonEmpty
    processingTimeMs: Millis
    data: dict[str, Any]


class ErrorResponse(BaseEnvelope):
    """standard error response body"""

    success: bool = False
    statusCode: int = Field(ge=ERROR_STATUS_MIN, le=ERROR_STATUS_MAX)
    errorCode: ErrorCode
    message: NonEmpty
    details: list[str] = Field(default_factory=list)
    modelName: str | None = None
    processingTimeMs: Millis | None = None

    @classmethod
    def from_code(
        cls,
        status_code: int,
        error_code: ErrorCode,
        request_id: str,
        details: list[str] | None = None,
        model_name: str | None = None,
        processing_time_ms: float | None = None,
    ) -> "ErrorResponse":
        return cls(
            statusCode=status_code,
            errorCode=error_code,
            message=get_error_message(error_code),
            details=details or [],
            requestId=request_id,
            modelName=model_name,
            processingTimeMs=processing_time_ms,
        )
