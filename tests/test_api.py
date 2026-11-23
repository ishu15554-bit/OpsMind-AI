
from fastapi.testclient import TestClient

from api.main import app
from src.database.db import get_prediction_history

# Initialize API test client
client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "Running"


def test_ticket_prediction():
    response = client.post(
        "/predict/ticket",
        json={
            "ticket_text": (
                "Unable to login to my account. "
                "Password reset is not working."
            )
        }
    )

    assert response.status_code == 200
    assert "prediction" in response.json()


def test_hdfs_anomaly_detection():
    response = client.post(
        "/predict/hdfs",
        json={
            "log_text": "blk_123 received packet from datanode"
        }
    )

    assert response.status_code == 200
    assert "prediction" in response.json()


def test_openstack_anomaly_detection():
    response = client.post(
        "/predict/openstack",
        json={
            "log_text": (
                "INFO nova.compute: Instance successfully started"
            )
        }
    )

    assert response.status_code == 200
    assert "prediction" in response.json()


def test_rag_assistant():
    response = client.post(
        "/rag/ask",
        json={
            "log_text": (
                "How can I troubleshoot a password reset issue?"
            )
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert data["answer"]


def test_prediction_database():
    history = get_prediction_history(limit=100)

    assert isinstance(history, list)


def test_empty_ticket_rejected():
    response = client.post(
        "/predict/ticket",
        json={"ticket_text": ""}
    )

    assert response.status_code == 422


def test_empty_log_rejected():
    response = client.post(
        "/predict/hdfs",
        json={"log_text": ""}
    )

    assert response.status_code == 422


def test_invalid_endpoint():
    response = client.get("/invalid-endpoint")

    assert response.status_code == 404
