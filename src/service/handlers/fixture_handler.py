from src.contracts import (
    RESULT_MODELS,
    FeatureResult,
    InferenceFeature,
    success_fixture,
)

from ..schemas import InferenceRequest


class FixtureHandler:
    """return sample json data for one feature"""

    def __init__(
        self,
        feature: InferenceFeature,
        model_suffix: str,
        model_version: str,
    ):
        self.feature = feature
        self.model_name = f"{feature.value}{model_suffix}"
        self.model_version = model_version
        self._result = RESULT_MODELS[feature].model_validate(
            success_fixture(feature.value)
        )

    async def predict(self, payload: InferenceRequest) -> FeatureResult:
        return self._result
