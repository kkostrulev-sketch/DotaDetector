from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"


def create_test_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (640, 480), color=(30, 60, 90))
    draw = ImageDraw.Draw(image)
    draw.rectangle((120, 180, 260, 320), fill=(200, 40, 40))
    draw.rectangle((360, 220, 520, 340), fill=(40, 180, 80))
    image.save(path, format="JPEG")


@pytest.fixture(scope="session")
def sample_image_path() -> Path:
    image_path = TEST_DATA_DIR / "sample.jpg"
    if not image_path.exists():
        create_test_image(image_path)
    return image_path


def _normalized_classes(classes: list[str]) -> set[str]:
    return {name.replace("-", " ").strip().lower() for name in classes}


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True

    classes = payload["classes"]
    normalized = _normalized_classes(classes)
    assert len(classes) == 15
    assert "plane" in normalized
    assert "ship" in normalized
    assert "small vehicle" in normalized
    assert "large vehicle" in normalized


def test_predict_endpoint(client: TestClient, sample_image_path: Path) -> None:
    with sample_image_path.open("rb") as image_file:
        response = client.post(
            "/predict",
            files={"file": ("sample.jpg", image_file, "image/jpeg")},
            params={"confidence": 0.1},
        )

    assert response.status_code == 200
    payload = response.json()
    assert "annotated_image_base64" in payload
    assert payload["image_width"] == 640
    assert payload["image_height"] == 480
    assert isinstance(payload["detections"], list)
    assert payload["latency_ms"] >= 0


def test_stats_and_history(client: TestClient, sample_image_path: Path) -> None:
    with sample_image_path.open("rb") as image_file:
        client.post(
            "/predict",
            files={"file": ("sample.jpg", image_file, "image/jpeg")},
        )

    stats_response = client.get("/stats")
    assert stats_response.status_code == 200
    stats = stats_response.json()
    assert stats["total_predictions"] >= 1

    history_response = client.get("/history")
    assert history_response.status_code == 200
    history = history_response.json()
    assert len(history) >= 1