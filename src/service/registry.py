from collections.abc import Iterable

from src.contracts import ErrorCode, InferenceFeature
from src.contracts.patterns import MODEL_NAME_PATTERN

from .errors.exceptions import ModelServiceError
from .handlers.base import ModelHandler


def _ensure_valid_name(handler: ModelHandler) -> None:
    if not MODEL_NAME_PATTERN.match(handler.model_name):
        raise ValueError(f"invalid model name {handler.model_name}")


class HandlerRegistry:
    """keep and find feature handlers"""

    def __init__(self, handlers: Iterable[ModelHandler]):
        self._handlers: dict[InferenceFeature, ModelHandler] = {}

        for handler in handlers:
            _ensure_valid_name(handler)
            self._handlers[handler.feature] = handler

    def resolve(self, feature: InferenceFeature) -> ModelHandler:
        handler = self._handlers.get(feature)

        if handler is None:
            raise ModelServiceError(
                ErrorCode.NOT_FOUND,
                details=[feature.value],
            )

        return handler

    @property
    def features(self) -> list[str]:
        return sorted(feature.value for feature in self._handlers)
