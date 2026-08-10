from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from api.app import app


client = TestClient(app)


def create_test_image():
    image = Image.new("RGB", (224, 224), color="white")

    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    return buffer


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "Cats vs Dogs Classification API"
    assert data["status"] == "running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict():
    image_buffer = create_test_image()

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.jpg",
                image_buffer,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["label"] in ("Cat", "Dog")
    assert "Cat" in data["probabilities"]
    assert "Dog" in data["probabilities"]
    assert "latency_ms" in data

    probability_sum = (
        data["probabilities"]["Cat"]
        + data["probabilities"]["Dog"]
    )

    assert abs(probability_sum - 1.0) < 1e-5
