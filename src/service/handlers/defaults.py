from src.contracts import InferenceFeature

from ..settings import ServiceSettings
from .base import ModelHandler
from .fixture_handler import FixtureHandler


def default_handlers(
    settings: ServiceSettings | None = None,
) -> list[ModelHandler]:
    cfg = settings or ServiceSettings()

    return [
        FixtureHandler(
            feature,
            model_suffix=cfg.fake_model_suffix,
            model_version=cfg.fake_model_version,
        )
        for feature in InferenceFeature
    ]
