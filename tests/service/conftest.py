import base64

import pytest
from fastapi.testclient import TestClient

from src.contracts import ErrorResponse
from src.service.main import create_app
from src.service.registry import HandlerRegistry
from src.service.settings import ServiceSettings
from support import StubHandler


@pytest.fixture
def make_client():
    def build(action=None, **settings):
        registry = HandlerRegistry([StubHandler(action)]) if action else None
        app = create_app(ServiceSettings(**settings), registry)

        return TestClient(app, raise_server_exceptions=False)

    return build


@pytest.fixture
def valid_body():
    raw = base64.b64encode(b"\xff\xd8\xffsample").decode("ascii")

    return {"imageBase64": f"data:image/jpeg;base64,{raw}"}


@pytest.fixture
def assert_error_shape():
    def check(response, status, code):
        body = response.json()

        assert response.status_code == status
        assert set(body) == set(ErrorResponse.model_fields)
        assert body["success"] is False
        assert body["errorCode"] == code
        assert body["requestId"] == response.headers["X-Request-ID"]

        return body

    return check
