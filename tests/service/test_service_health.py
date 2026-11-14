import uuid

from src.contracts import InferenceFeature, contract_version
from src.service.routes.paths import HEALTH_PATH
from src.service.runtime.context import REQUEST_ID_HEADER

SAMPLE_REQUEST_ID = "req-123"
BAD_REQUEST_ID = "bad id with spaces"


def test_health_reports_service_and_all_features(make_client):
    body = make_client().get(HEALTH_PATH).json()

    assert body["status"] == "ok"
    assert body["features"] == sorted(f.value for f in InferenceFeature)


def test_health_reports_contract_version(make_client):
    body = make_client().get(HEALTH_PATH).json()

    assert body["contractVersion"] == contract_version()


def test_request_id_is_echoed(make_client):
    response = make_client().get(
        HEALTH_PATH, headers={REQUEST_ID_HEADER: SAMPLE_REQUEST_ID}
    )

    assert response.headers[REQUEST_ID_HEADER] == SAMPLE_REQUEST_ID


def test_uuid_request_id_is_echoed(make_client):
    request_id = str(uuid.uuid4())
    response = make_client().get(
        HEALTH_PATH, headers={REQUEST_ID_HEADER: request_id}
    )

    assert response.headers[REQUEST_ID_HEADER] == request_id


def test_unsafe_request_id_is_replaced(make_client):
    response = make_client().get(
        HEALTH_PATH, headers={REQUEST_ID_HEADER: BAD_REQUEST_ID}
    )

    assert response.headers[REQUEST_ID_HEADER] != BAD_REQUEST_ID
