"""Gradio GUI for DOTA object detection on images and video."""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import gradio as gr
import numpy as np
from PIL import Image

from service.app.config import load_settings
from service.app.database import HistoryStore
from service.app.detector import DDetector

settings = load_settings()
detector = DDetector(settings)
history_store = HistoryStore(settings.history_db, settings.results_dir)


def _detections_to_table(detections: list) -> list[list]:
    return [
        [
            item.class_name,
            f"{item.confidence:.2%}",
            f"({item.x1:.0f}, {item.y1:.0f})",
            f"({item.x2:.0f}, {item.y2:.0f})",
        ]
        for item in detections
    ]


def _annotated_numpy_from_result(result) -> np.ndarray:
    image_bytes = base64.b64decode(result.annotated_image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image)


def _save_image_history(filename: str, result, confidence: float) -> None:
    detections_payload = [
        {
            "class_id": item.class_id,
            "class_name": item.class_name,
            "confidence": round(item.confidence, 4),
            "x1": round(item.x1, 2),
            "y1": round(item.y1, 2),
            "x2": round(item.x2, 2),
            "y2": round(item.y2, 2),
        }
        for item in result.detections
    ]
    annotated_bytes = base64.b64decode(result.annotated_image_base64)
    history_store.add_prediction(
        filename=filename,
        model_version=settings.model_version,
        detections=detections_payload,
        latency_ms=result.latency_ms,
        confidence_threshold=confidence,
        annotated_image_bytes=annotated_bytes,
    )


def detect_image(image: np.ndarray | None, confidence: float) -> tuple:
    if image is None:
        return None, [], "Загрузите изображение."

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG")
    result = detector.predict(buffer.getvalue(), confidence=confidence)

    _save_image_history("gradio_upload.jpg", result, confidence)

    summary = (
        f"Найдено объектов: {len(result.detections)}\n"
        f"Задержка: {result.latency_ms:.1f} мс\n"
        f"Размер: {result.image_width}×{result.image_height}"
    )
    return _annotated_numpy_from_result(result), _detections_to_table(result.detections), summary


def detect_video(video_path: str | tuple | None, confidence: float) -> tuple:
    if isinstance(video_path, tuple):
        video_path = video_path[0]

    if not video_path:
        return None, "Загрузите видеофайл (MP4, AVI)."

    path = Path(str(video_path))
    if not path.exists():
        return None, f"Файл не найден: {video_path}"

    result = detector.predict_video(path, confidence=confidence)
    history_store.add_prediction(
        filename=path.name,
        model_version=settings.model_version,
        detections=[],
        latency_ms=result.latency_ms,
        confidence_threshold=confidence,
        annotated_image_bytes=None,
    )

    summary = (
        f"Кадров обработано: {result.frames_processed}\n"
        f"Всего детекций: {result.total_detections}\n"
        f"Средний FPS: {result.average_fps}\n"
        f"Время обработки: {result.latency_ms:.1f} мс\n"
        f"Файл: {result.output_path.name}"
    )
    return str(result.output_path), summary


def load_model_info() -> tuple[str, str]:
    classes = detector.get_class_names_list()
    stats = history_store.get_stats()

    model_info = (
        f"Модель: {settings.model_version}\n"
        f"Архитектура: {settings.architecture}\n"
        f"Классов: {len(classes)}\n"
        f"Порог по умолчанию: {settings.confidence_threshold}\n\n"
        f"Классы:\n" + ", ".join(classes)
    )

    history_rows = history_store.list_predictions(limit=10)
    if not history_rows:
        history_text = "История пуста."
    else:
        lines = []
        for item in history_rows:
            lines.append(
                f"#{item['id']} · {item['filename']} · "
                f"{item['detections_count']} obj · {item['latency_ms']} ms · "
                f"{item['created_at'][:19]}"
            )
        history_text = "\n".join(lines)

    stats_text = (
        f"Всего запросов: {stats['total_predictions']}\n"
        f"Средняя задержка: {stats['average_latency_ms']} мс\n"
        f"Среднее число объектов: {stats['average_detections']}\n"
        f"Ошибок: {stats['error_count']}"
    )

    return model_info, f"{stats_text}\n\nПоследние запуски:\n{history_text}"


GRADIO_THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="green",
    neutral_hue="slate",
).set(
    body_background_fill="*neutral_950",
    block_background_fill="*neutral_900",
    block_border_color="*neutral_700",
    body_text_color="*neutral_100",
)
GRADIO_CSS = ".gradio-container {max-width: 1200px !important}"


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="DOTA Detector") as demo:
        gr.Markdown(
            """
            # DOTA Detector · YOLOv8 OBB
            Детекция объектов на аэрофотоснимках и видео (датасет DOTAv1).
            """
        )

        with gr.Tab("Изображение"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(label="Загрузите изображение", type="numpy")
                    image_confidence = gr.Slider(0.05, 0.95, value=settings.confidence_threshold, step=0.05, label="Порог уверенности")
                    image_button = gr.Button("Найти объекты", variant="primary")
                with gr.Column():
                    image_output = gr.Image(label="Результат с bounding boxes")
                    image_summary = gr.Textbox(label="Сводка", lines=4)
            image_table = gr.Dataframe(
                headers=["Класс", "Уверенность", "Верхний левый", "Нижний правый"],
                label="Детекции",
                interactive=False,
            )
            image_button.click(
                detect_image,
                inputs=[image_input, image_confidence],
                outputs=[image_output, image_table, image_summary],
            )

        with gr.Tab("Видео"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Загрузите видео")
                    video_confidence = gr.Slider(0.05, 0.95, value=settings.confidence_threshold, step=0.05, label="Порог уверенности")
                    video_button = gr.Button("Обработать видео", variant="primary")
                with gr.Column():
                    video_output = gr.Video(label="Аннотированное видео")
                    video_summary = gr.Textbox(label="Сводка", lines=6)
            video_button.click(
                detect_video,
                inputs=[video_input, video_confidence],
                outputs=[video_output, video_summary],
            )

        with gr.Tab("Модель и история"):
            refresh_button = gr.Button("Обновить")
            model_info = gr.Textbox(label="Информация о модели", lines=12)
            history_info = gr.Textbox(label="Статистика и история", lines=14)
            demo.load(load_model_info, outputs=[model_info, history_info])
            refresh_button.click(load_model_info, outputs=[model_info, history_info])

    return demo


def main() -> None:
    demo = build_demo()
    print("Gradio: откройте http://localhost:7860")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=GRADIO_THEME,
        css=GRADIO_CSS,
    )


if __name__ == "__main__":
    main()