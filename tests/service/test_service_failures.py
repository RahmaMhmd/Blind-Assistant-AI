import asyncio
from http import HTTPStatus

from src.contracts import ErrorCode
from tests.service.support import OCR_PATH, StubHandler, stub_result


def test_timeout_has_its_own_code(make_client, valid_body, assert_error_shape):
    async def slow():
        await asyncio.sleep(1)

    client = make_client(slow, request_timeout_seconds=0.05)
    response = client.post(OCR_PATH, json=valid_body)
    body = assert_error_shape(
        response, HTTPStatus.GATEWAY_TIMEOUT, ErrorCode.REQUEST_TIMEOUT.value
    )

    assert body["modelName"] == StubHandler.model_name


def test_low_confidence_has_own_code(make_client, valid_body, assert_error_shape):
    async def unsure():
        return stub_result(0.1)

    response = make_client(unsure).post(OCR_PATH, json=valid_body)
    body = assert_error_shape(
        response, HTTPStatus.UNPROCESSABLE_ENTITY, ErrorCode.LOW_CONFIDENCE.value
    )

    assert body["modelName"] == StubHandler.model_name


def test_confident_result_passes(make_client, valid_body):
    async def sure():
        return stub_result(0.9)

    response = make_client(sure).post(OCR_PATH, json=valid_body)

    assert response.status_code == HTTPStatus.OK


def test_threshold_is_configurable(make_client, valid_body):
    async def middle():
        return stub_result(0.5)

    client = make_client(middle, min_confidence=0.4)

    assert client.post(OCR_PATH, json=valid_body).status_code == HTTPStatus.OK


def test_unexpected_error_hides_internals(
    make_client, valid_body, assert_error_shape
):
    async def broken():
        raise RuntimeError("secret path /srv/model.onnx")

    response = make_client(broken).post(OCR_PATH, json=valid_body)
    body = assert_error_shape(
        response,
        HTTPStatus.INTERNAL_SERVER_ERROR,
        ErrorCode.INTERNAL_SERVER_ERROR.value,
    )

    assert body["details"] == []
    assert "secret" not in response.text
