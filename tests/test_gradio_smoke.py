from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def test_gradio_app_imports() -> None:
    from service import gradio_app

    assert hasattr(gradio_app, "detect_image")
    assert hasattr(gradio_app, "detect_video")
    assert hasattr(gradio_app, "build_demo")


def test_detect_image_on_real_sample() -> None:
    from service.gradio_app import detect_image

    image_path = Path(__file__).resolve().parent / "test_data" / "P0723.png"
    image = np.array(Image.open(image_path).convert("RGB"))

    annotated, table, summary = detect_image(image, confidence=0.25)

    assert annotated is not None
    assert annotated.shape[0] > 0
    assert len(table) >= 1
    assert "Найдено объектов" in summary


def test_detect_video_requires_file() -> None:
    from service.gradio_app import detect_video

    output, summary = detect_video(None, confidence=0.25)
    assert output is None
    assert "Загрузите" in summary