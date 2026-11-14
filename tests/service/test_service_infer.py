from http import HTTPStatus

import pytest

from src.contracts import ErrorCode, InferenceFeature
from src.service.registry import HandlerRegistry
from src.service.runtime.context import REQUEST_ID_HEADER
from src.service.settings import ServiceSettings
from tests.helpers import image_body
from tests.service.support import OCR_PATH, StubHandler, feature_path, stub_result


def test_ocr_returns_success_shape(make_client, valid_body):
    settings = ServiceSettings()
    response = make_client().post(OCR_PATH, json=valid_body)
    body = response.json()

    assert response.status_code == HTTPStatus.OK
    assert body["success"] is True
    assert body["modelName"] == f"{InferenceFeature.OCR.value}{settings.fake_model_suffix}"
    assert body["modelVersion"] == settings.fake_model_version
    assert body["requestId"] == response.headers[REQUEST_ID_HEADER]
    assert body["processingTimeMs"] >= 0
    assert "extractedText" in body["data"]


def test_missing_image_is_bad_input(make_client, assert_error_shape):
    response = make_client().post(OCR_PATH, json={})

    assert_error_shape(
        response, HTTPStatus.BAD_REQUEST, ErrorCode.VALIDATION_FAILED.value
    )


def test_blank_image_is_bad_input(make_client, assert_error_shape):
    response = make_client().post(OCR_PATH, json={"imageBase64": "   "})

    assert_error_shape(
        response, HTTPStatus.BAD_REQUEST, ErrorCode.VALIDATION_FAILED.value
    )


def test_unknown_field_is_bad_input(make_client, valid_body, assert_error_shape):
    response = make_client().post(OCR_PATH, json={**valid_body, "extra": 1})

    assert_error_shape(
        response, HTTPStatus.BAD_REQUEST, ErrorCode.VALIDATION_FAILED.value
    )


def test_unsupported_image_keeps_contract_code(make_client, assert_error_shape):
    body = image_body("image/webp", b"data")
    response = make_client().post(OCR_PATH, json=body)

    assert_error_shape(
        response,
        HTTPStatus.UNPROCESSABLE_ENTITY,
        ErrorCode.UNSUPPORTED_IMAGE_FORMAT.value,
    )


def test_unserved_feature_is_not_found(make_client, valid_body, assert_error_shape):
    async def sure():
        return stub_result(0.9)

    client = make_client(sure)
    path = feature_path(InferenceFeature.SCENE_DESCRIPTION)
    response = client.post(path, json=valid_body)

    assert_error_shape(response, HTTPStatus.NOT_FOUND, ErrorCode.NOT_FOUND.value)


def test_unknown_route_uses_same_shape(make_client, assert_error_shape):
    response = make_client().get("/missing")

    assert_error_shape(response, HTTPStatus.NOT_FOUND, ErrorCode.NOT_FOUND.value)


def test_registry_rejects_invalid_model_name():
    handler = StubHandler(lambda: stub_result(0.9))
    handler.model_name = "Bad Model!"

    with pytest.raises(ValueError):
        HandlerRegistry([handler])
