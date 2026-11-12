import asyncio

from support import OCR_PATH, stub_result


def test_timeout_maps_to_service_unavailable(
    make_client, valid_body, assert_error_shape
):
    async def slow():
        await asyncio.sleep(1)

    client = make_client(slow, request_timeout_seconds=0.05)
    response = client.post(OCR_PATH, json=valid_body)
    body = assert_error_shape(response, 503, "ERR_SERVICE_UNAVAILABLE")

    assert body["modelName"] == "stub"


def test_low_confidence_has_own_code(
    make_client, valid_body, assert_error_shape
):
    async def unsure():
        return stub_result(0.1)

    response = make_client(unsure).post(OCR_PATH, json=valid_body)
    body = assert_error_shape(response, 422, "ERR_LOW_CONFIDENCE")

    assert body["modelName"] == "stub"


def test_confident_result_passes(make_client, valid_body):
    async def sure():
        return stub_result(0.9)

    response = make_client(sure).post(OCR_PATH, json=valid_body)

    assert response.status_code == 200


def test_threshold_is_configurable(make_client, valid_body):
    async def middle():
        return stub_result(0.5)

    client = make_client(middle, min_confidence=0.4)

    assert client.post(OCR_PATH, json=valid_body).status_code == 200


def test_unexpected_error_hides_internals(
    make_client, valid_body, assert_error_shape
):
    async def broken():
        raise RuntimeError("secret path /srv/model.onnx")

    response = make_client(broken).post(OCR_PATH, json=valid_body)
    body = assert_error_shape(response, 500, "ERR_INTERNAL_SERVER_ERROR")

    assert body["details"] == []
    assert "secret" not in response.text
