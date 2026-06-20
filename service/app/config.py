from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


def resolve_device(requested: str) -> str:
    normalized = requested.strip().lower()
    if normalized in {"", "auto"}:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if normalized == "cuda":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return normalized

SERVICE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = SERVICE_ROOT / "configs" / "inference.yaml"


@dataclass(frozen=True)
class Settings:
    weights_path: Path
    model_version: str
    architecture: str
    input_size: int
    classes: list[str]
    confidence_threshold: float
    iou_threshold: float
    device: str
    host: str
    port: int
    max_upload_size_mb: int
    history_db: Path
    results_dir: Path


def load_settings() -> Settings:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)

    model_cfg = raw["model"]
    inference_cfg = raw["inference"]
    service_cfg = raw["service"]

    weights_path = (CONFIG_PATH.parent / model_cfg["weights_path"]).resolve()
    history_db = (SERVICE_ROOT / service_cfg["history_db"]).resolve()
    results_dir = (SERVICE_ROOT / service_cfg["results_dir"]).resolve()

    return Settings(
        weights_path=weights_path,
        model_version=model_cfg["version"],
        architecture=model_cfg["architecture"],
        input_size=model_cfg["input_size"],
        classes=model_cfg["classes"],
        confidence_threshold=float(inference_cfg["confidence_threshold"]),
        iou_threshold=float(inference_cfg["iou_threshold"]),
        device=resolve_device(inference_cfg["device"]),
        host=service_cfg["host"],
        port=int(service_cfg["port"]),
        max_upload_size_mb=int(service_cfg["max_upload_size_mb"]),
        history_db=history_db,
        results_dir=results_dir,
    )