import pytest

from src.contracts import (
    RESULT_MODELS,
    ErrorResponse,
    FacialExpression,
    HapticPattern,
    InferenceFeature,
    MobileAction,
    VoiceIntent,
    success_fixture,
)
from src.contracts.catalog import (
    CONTRACTS_DIR,
    ENUMS_FILE,
    ERROR_FIXTURES_DIR,
    read_json,
)

UNKNOWN_CODE_FIXTURE = "unknown_future_code.json"
ENUM_CLASSES = (VoiceIntent, MobileAction, FacialExpression, HapticPattern)


@pytest.mark.parametrize("enum_class", ENUM_CLASSES)
def test_enum_values_match_contract(enum_class):
    expected = read_json(ENUMS_FILE)[enum_class.__name__]

    assert [member.value for member in enum_class] == expected


@pytest.mark.parametrize("feature", list(InferenceFeature))
def test_success_fixture_fits_result_model(feature):
    model = RESULT_MODELS[feature]
    parsed = model.model_validate(success_fixture(feature.value))

    assert parsed.model_dump(mode="json") == success_fixture(feature.value)


def test_error_fixtures_fit_error_response():
    folder = CONTRACTS_DIR / ERROR_FIXTURES_DIR
    names = [
        path.name
        for path in folder.glob("*.json")
        if path.name != UNKNOWN_CODE_FIXTURE
    ]

    assert names
    for name in names:
        ErrorResponse.model_validate(read_json(f"{ERROR_FIXTURES_DIR}/{name}"))
