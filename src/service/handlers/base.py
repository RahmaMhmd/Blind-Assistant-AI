from typing import Protocol

from src.contracts import FeatureResult, InferenceFeature

from ..schemas import InferenceRequest


class ModelHandler(Protocol):
    feature: InferenceFeature
    model_name: str
    model_version: str

    async def predict(self, payload: InferenceRequest) -> FeatureResult: ...
