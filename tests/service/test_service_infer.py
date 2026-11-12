from support import OCR_PATH


def test_fake_ocr_returns_success_shape(make_client, valid_body):
    response = make_client().post(OCR_PATH, json=valid_body)
    body = response.json()

    assert response.status_code == 200
    assert body["success"] is True
    assert body["modelName"] == "fake-ocr"
    assert body["requestId"] == response.headers["X-Request-ID"]
    assert body["processingTimeMs"] >= 0
    assert "extractedText" in body["data"]


def test_missing_image_is_bad_input(
    make_client, assert_error_shape
):
    response = make_client().post(OCR_PATH, json={})

    assert_error_shape(response, 400, "ERR_VALIDATION_FAILED")


def test_unknown_field_is_bad_input(
    make_client, valid_body, assert_error_shape
):
    response = make_client().post(OCR_PATH, json={**valid_body, "extra": 1})

    assert_error_shape(response, 400, "ERR_VALIDATION_FAILED")


def test_unsupported_image_keeps_contract_code(
    make_client, assert_error_shape
):
    body = {"imageBase64": "data:image/webp;base64,AAAA"}
    response = make_client().post(OCR_PATH, json=body)

    assert_error_shape(response, 422, "ERR_UNSUPPORTED_IMAGE_FORMAT")


def test_unserved_feature_is_not_found(
    make_client, valid_body, assert_error_shape
):
    response = make_client().post(
        "/v1/infer/scene_description", json=valid_body
    )

    assert_error_shape(response, 404, "ERR_NOT_FOUND")


def test_unknown_route_uses_same_shape(make_client, assert_error_shape):
    response = make_client().get("/missing")

    assert_error_shape(response, 404, "ERR_NOT_FOUND")
