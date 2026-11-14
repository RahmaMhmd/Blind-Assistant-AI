from http import HTTPStatus

import pytest

from src.contracts import InferenceFeature, success_fixture
from src.contracts.features import FEATURE_INPUT_TYPES, InputKind
from src.service.settings import ServiceSettings
from tests.service.support import feature_path

SPEECH_BODY = {"userSpeech": "read this paper"}


@pytest.mark.parametrize("feature", list(InferenceFeature))
def test_every_feature_returns_its_fixture(make_client, valid_body, feature):
    settings = ServiceSettings()
    is_image = FEATURE_INPUT_TYPES[feature] is InputKind.IMAGE
    body = valid_body if is_image else SPEECH_BODY

    response = make_client().post(feature_path(feature), json=body)
    payload = response.json()

    assert response.status_code == HTTPStatus.OK
    assert payload["modelName"] == f"{feature.value}{settings.fake_model_suffix}"
    assert payload["modelVersion"] == settings.fake_model_version
    assert payload["data"] == success_fixture(feature.value)
