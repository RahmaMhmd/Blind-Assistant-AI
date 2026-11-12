from typing import Protocol

from pydantic import BaseModel

from src.contracts import InferenceFeature

from ..schemas import InferenceRequest


class ModelHandler(Protocol):
    feature: InferenceFeature
    model_name: str

    async def predict(self, payload: InferenceRequest) -> BaseModel: ...
