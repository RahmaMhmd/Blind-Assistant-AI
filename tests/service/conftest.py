import pytest
from fastapi.testclient import TestClient

from src.contracts import ErrorResponse
from src.service.main import create_app
from src.service.registry import HandlerRegistry
from src.service.runtime.context import REQUEST_ID_HEADER
from src.service.settings import ServiceSettings
from tests.helpers import image_body
from tests.service.support import StubHandler


@pytest.fixture
def make_client():
    def build(action=None, **settings):
        registry = HandlerRegistry([StubHandler(action)]) if action else None
        app = create_app(ServiceSettings(**settings), registry)

        return TestClient(app, raise_server_exceptions=False)

    return build


@pytest.fixture
def valid_body():
    return image_body()


@pytest.fixture
def assert_error_shape():
    def check(response, status, code):
        body = response.json()

        assert response.status_code == status
        assert set(body) == set(ErrorResponse.model_fields)
        assert body["success"] is False
        assert body["errorCode"] == code
        assert body["requestId"] == response.headers[REQUEST_ID_HEADER]

        return body

    return check
