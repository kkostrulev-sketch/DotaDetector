from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"

REAL_IMAGE_NAMES = ["P0723.png", "P0008.png", "P1928.png"]

REAL_IMAGES = [
    pytest.param(filename, id=Path(filename).stem)
    for filename in REAL_IMAGE_NAMES
]


@pytest.mark.parametrize("filename", REAL_IMAGES)
def test_predict_on_real_drone_image(client: TestClient, filename: str) -> None:
    image_path = TEST_DATA_DIR / filename
    assert image_path.exists(), f"Тестовое изображение не найдено: {image_path}"

    with image_path.open("rb") as image_file:
        response = client.post(
            "/predict",
            files={"file": (filename, image_file, "image/png")},
            params={"confidence": 0.25},
        )

    assert response.status_code == 200
    payload = response.json()

    assert payload["filename"] == filename
    assert payload["detections_count"] == len(payload["detections"])
    assert payload["image_width"] > 0
    assert payload["image_height"] > 0
    assert payload["latency_ms"] > 0
    assert len(payload["annotated_image_base64"]) > 100
    assert payload["detections_count"] >= 1, (
        f"{filename}: ожидалась хотя бы 1 детекция, получено {payload['detections_count']}"
    )


@pytest.mark.parametrize("filename", REAL_IMAGES)
def test_detector_direct_on_real_images(filename: str) -> None:
    from service.app.config import load_settings
    from service.app.detector import DDetector

    image_path = TEST_DATA_DIR / filename
    settings = load_settings()
    detector = DDetector(settings)
    result = detector.predict(image_path.read_bytes(), confidence=0.25)

    assert len(result.detections) >= 1, (
        f"{filename}: ожидалась хотя бы 1 детекция, получено {len(result.detections)}"
    )
    assert result.latency_ms > 0
    assert len(result.annotated_image_base64) > 100


def test_batch_predict_on_all_real_images(client: TestClient) -> None:
    files = []
    for filename in REAL_IMAGE_NAMES:
        image_path = TEST_DATA_DIR / filename
        files.append(
            ("files", (filename, image_path.read_bytes(), "image/png")),
        )

    response = client.post("/batch_predict", files=files, params={"confidence": 0.25})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == len(REAL_IMAGE_NAMES)

    total_detections = sum(item["detections_count"] for item in payload)
    assert total_detections >= len(REAL_IMAGE_NAMES)

    filenames = {item["filename"] for item in payload}
    assert filenames == set(REAL_IMAGE_NAMES)