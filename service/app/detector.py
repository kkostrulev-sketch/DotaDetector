from __future__ import annotations

import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import base64
import io
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

from service.app.config import Settings

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class PredictionResult:
    detections: list[Detection]
    annotated_image_base64: str
    latency_ms: float
    image_width: int
    image_height: int


@dataclass
class VideoPredictionResult:
    output_path: Path
    total_detections: int
    frames_processed: int
    latency_ms: float
    average_fps: float


class DDetector:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = YOLO(str(settings.weights_path))
        self.class_names = self.model.names
        # Определяем тип модели: OBB или обычная
        self.is_obb = "obb" in str(settings.weights_path).lower() or "obb" in settings.architecture.lower()

    def get_class_names_list(self) -> list[str]:
        if isinstance(self.class_names, dict):
            return [str(self.class_names[key]) for key in sorted(self.class_names)]
        return [str(name) for name in self.class_names]

    def predict(self, image_bytes: bytes, confidence: float | None = None) -> PredictionResult:
        started = time.perf_counter()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_width, image_height = image.size

        conf = confidence if confidence is not None else self.settings.confidence_threshold
        results = self.model.predict(
            source=np.array(image),
            conf=conf,
            iou=self.settings.iou_threshold,
            imgsz=self.settings.input_size,
            device=self.settings.device,
            verbose=False,
        )

        detections: list[Detection] = []
        annotated = image

        if results:
            result = results[0]

            # Проверяем тип модели: OBB или обычная
            if self.is_obb and hasattr(result, 'obb') and result.obb is not None:
                # OBB модель - используем ориентированные bounding boxes
                obb_boxes = result.obb
                for box in obb_boxes:
                    class_id = int(box.cls.item())
                    # Для OBB получаем горизонтальный bbox (xyxy), который описывает повёрнутый bbox
                    xyxy = box.xyxy[0]
                    detections.append(
                        Detection(
                            class_id=class_id,
                            class_name=str(self.class_names[class_id]),
                            confidence=float(box.conf.item()),
                            x1=float(xyxy[0].item()),
                            y1=float(xyxy[1].item()),
                            x2=float(xyxy[2].item()),
                            y2=float(xyxy[3].item()),
                        )
                    )
            elif hasattr(result, 'boxes') and result.boxes is not None:
                # Обычная модель (HBB)
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls.item())
                    detections.append(
                        Detection(
                            class_id=class_id,
                            class_name=str(self.class_names[class_id]),
                            confidence=float(box.conf.item()),
                            x1=float(box.xyxy[0][0].item()),
                            y1=float(box.xyxy[0][1].item()),
                            x2=float(box.xyxy[0][2].item()),
                            y2=float(box.xyxy[0][3].item()),
                        )
                    )

            # Визуализация работает для обоих типов моделей
            plotted = result.plot()
            annotated = Image.fromarray(plotted[..., ::-1])

        buffer = io.BytesIO()
        annotated.save(buffer, format="JPEG", quality=90)
        annotated_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        latency_ms = (time.perf_counter() - started) * 1000

        return PredictionResult(
            detections=detections,
            annotated_image_base64=annotated_b64,
            latency_ms=latency_ms,
            image_width=image_width,
            image_height=image_height,
        )

    def _count_detections_in_result(self, result) -> int:
        if self.is_obb and hasattr(result, "obb") and result.obb is not None:
            return len(result.obb)
        if hasattr(result, "boxes") and result.boxes is not None:
            return len(result.boxes)
        return 0

    def predict_video(self, video_path: Path, confidence: float | None = None) -> VideoPredictionResult:
        started = time.perf_counter()
        conf = confidence if confidence is not None else self.settings.confidence_threshold

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        run_name = f"gradio_{timestamp}"
        output_dir = self.settings.results_dir / "videos" / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        results = self.model.predict(
            source=str(video_path.resolve()),
            conf=conf,
            iou=self.settings.iou_threshold,
            imgsz=self.settings.input_size,
            device=self.settings.device,
            save=True,
            project=str(output_dir.parent),
            name=run_name,
            exist_ok=True,
            verbose=False,
            stream=True,
        )

        total_detections = 0
        frames_processed = 0
        for result in results:
            frames_processed += 1
            total_detections += self._count_detections_in_result(result)

        latency_ms = (time.perf_counter() - started) * 1000
        average_fps = frames_processed / (latency_ms / 1000) if latency_ms > 0 else 0.0

        saved_videos = sorted(output_dir.rglob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not saved_videos:
            saved_videos = sorted(output_dir.rglob("*.avi"), key=lambda path: path.stat().st_mtime, reverse=True)

        if saved_videos:
            output_path = saved_videos[0]
        else:
            fallback = self.settings.results_dir / "videos" / f"{run_name}_{video_path.stem}.mp4"
            output_path = fallback
            if not output_path.exists():
                raise FileNotFoundError("Не удалось сохранить аннотированное видео")

        return VideoPredictionResult(
            output_path=output_path.resolve(),
            total_detections=total_detections,
            frames_processed=frames_processed,
            latency_ms=latency_ms,
            average_fps=round(average_fps, 2),
        )