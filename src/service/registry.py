from collections.abc import Iterable

from src.contracts import ErrorCode, InferenceFeature

from .errors.exceptions import ModelServiceError
from .handlers.base import ModelHandler


class HandlerRegistry:
    def __init__(self, handlers: Iterable[ModelHandler]):
        self._handlers = {handler.feature: handler for handler in handlers}

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
