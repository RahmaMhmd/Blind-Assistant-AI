import logging
from http import HTTPStatus

from src.contracts import ErrorCode
from src.contracts.image_rules import DEFAULT_MAX_IMAGE_BYTES
from src.service.settings import ServiceSettings
from tests.helpers import sized_jpeg_body
from tests.service.support import OCR_PATH


def test_default_limit_comes_from_settings():
    assert ServiceSettings().max_image_bytes == DEFAULT_MAX_IMAGE_BYTES


def test_image_over_default_limit_is_rejected(make_client, assert_error_shape):
    body = sized_jpeg_body(DEFAULT_MAX_IMAGE_BYTES + 1)
    response = make_client().post(OCR_PATH, json=body)

    assert_error_shape(
        response,
        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        ErrorCode.FILE_TOO_LARGE.value,
    )


def test_limit_follows_settings(make_client, assert_error_shape):
    client = make_client(max_image_bytes=64)
    response = client.post(OCR_PATH, json=sized_jpeg_body(65))

    assert_error_shape(
        response,
        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        ErrorCode.FILE_TOO_LARGE.value,
    )
    assert client.post(OCR_PATH, json=sized_jpeg_body(64)).status_code == HTTPStatus.OK


def test_unhandled_error_log_hides_payload(make_client, valid_body, caplog):
    async def broken():
        raise ValueError(valid_body["imageBase64"])

    with caplog.at_level(logging.DEBUG):
        make_client(broken).post(OCR_PATH, json=valid_body)

    assert valid_body["imageBase64"] not in caplog.text
