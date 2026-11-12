from src.service.runtime.context import REQUEST_ID_HEADER


def test_health_reports_service_and_features(make_client):
    response = make_client().get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["features"] == ["ocr"]


def test_request_id_is_echoed(make_client):
    response = make_client().get(
        "/health", headers={REQUEST_ID_HEADER: "req-123"}
    )

    assert response.headers[REQUEST_ID_HEADER] == "req-123"


def test_unsafe_request_id_is_replaced(make_client):
    response = make_client().get(
        "/health", headers={REQUEST_ID_HEADER: "bad id with spaces"}
    )

    assert response.headers[REQUEST_ID_HEADER] != "bad id with spaces"
