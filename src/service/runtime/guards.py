import asyncio
from collections.abc import Awaitable
from typing import TypeVar

from src.contracts import ErrorCode, FeatureResult

from ..errors.exceptions import ModelServiceError

T = TypeVar("T")


async def with_deadline(
    work: Awaitable[T],
    seconds: float,
    model_name: str,
) -> T:
    """stop model work after time limit"""
    try:
        return await asyncio.wait_for(work, timeout=seconds)
    except TimeoutError:
        raise ModelServiceError(
            ErrorCode.REQUEST_TIMEOUT,
            model_name=model_name,
        ) from None


def ensure_confident(
    result: FeatureResult,
    minimum: float,
    model_name: str,
) -> None:
    """reject result under confidence limit"""
    if result.confidence < minimum:
        raise ModelServiceError(
            ErrorCode.LOW_CONFIDENCE,
            model_name=model_name,
        )
