import asyncio
from collections.abc import Awaitable
from typing import TypeVar

from pydantic import BaseModel

from src.contracts import ErrorCode

from ..errors.exceptions import ModelServiceError

T = TypeVar("T")


async def with_deadline(
    work: Awaitable[T],
    seconds: float,
    model_name: str,
) -> T:
    try:
        return await asyncio.wait_for(work, timeout=seconds)
    except TimeoutError:
        raise ModelServiceError(
            ErrorCode.SERVICE_UNAVAILABLE,
            model_name=model_name,
        ) from None


def ensure_confident(
    result: BaseModel,
    minimum: float,
    model_name: str,
) -> None:
    if result.confidence < minimum:
        raise ModelServiceError(
            ErrorCode.LOW_CONFIDENCE,
            model_name=model_name,
        )
