from __future__ import annotations

from typing import Optional  # <--- Добавлен импорт Optional

from pydantic import BaseModel, ConfigDict, Field  # <--- Добавлен импорт ConfigDict


class DetectionSchema(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class PredictResponse(BaseModel):
    # <--- Отключаем защиту namespace "model_", чтобы поле model_version не вызывало warning
    model_config = ConfigDict(protected_namespaces=())

    id: int
    filename: str
    model_version: str
    architecture: str
    detections: list[DetectionSchema]
    detections_count: int
    latency_ms: float
    image_width: int
    image_height: int
    annotated_image_base64: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_version: str
    architecture: str
    classes: list[str]


class StatsResponse(BaseModel):
    total_predictions: int
    average_latency_ms: float
    average_detections: float
    error_count: int


class HistoryItem(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: int
    created_at: str
    filename: str
    model_version: str
    detections_count: int
    latency_ms: float
    confidence_threshold: float
    detections: list[DetectionSchema] = Field(default_factory=list)

    # <--- Заменено str | None на Optional[str]
    result_image_path: Optional[str] = None
    error: Optional[str] = None