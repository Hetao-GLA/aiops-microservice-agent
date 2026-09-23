from fastapi.testclient import TestClient
import pytest

import app.main as main


@pytest.fixture(autouse=True)
def disable_database_initialisation(monkeypatch):
    monkeypatch.setattr(main, "initialise_database", lambda: None)


def test_liveness() -> None:
    with TestClient(main.app) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_database_health_when_healthy(monkeypatch) -> None:
    monkeypatch.setattr(main, "check_database", lambda: (True, "healthy"))

    with TestClient(main.app) as client:
        response = client.get("/health/database")

    assert response.status_code == 200
    assert response.json()["component"] == "database"


def test_database_health_when_unhealthy(monkeypatch) -> None:
    monkeypatch.setattr(main, "check_database", lambda: (False, "OperationalError"))

    with TestClient(main.app) as client:
        response = client.get("/health/database")

    assert response.status_code == 503
    assert response.json()["error_type"] == "OperationalError"
