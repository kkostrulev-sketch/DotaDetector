"""Evaluate a trained detector and export detection metrics."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "metrics"


def evaluate_model(
    weights: Path,
    data_yaml: Path,
    imgsz: int,
    device: str,
    confidence: float,
    iou: float,
) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights.resolve()))

    started = time.perf_counter()
    metrics = model.val(
        data=str(data_yaml.resolve()),
        imgsz=imgsz,
        device=device,
        conf=confidence,
        iou=iou,
        verbose=False,
    )
    latency_ms = (time.perf_counter() - started) * 1000

    box = metrics.box
    return {
        "weights": str(weights.resolve()),
        "data": str(data_yaml.resolve()),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "latency_ms": round(latency_ms, 2),
        "metrics": {
            "mAP50": round(float(box.map50), 4),
            "mAP50-95": round(float(box.map), 4),
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
        },
        "per_class_ap50": {
            str(name): round(float(ap50), 4)
            for name, ap50 in zip(metrics.names.values(), box.ap50)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DOTA detector metrics")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "weights" / "yolov8n-obb.pt")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "DOTAv1.yaml")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "latest_eval.json")
    args = parser.parse_args()

    payload = evaluate_model(
        weights=args.weights,
        data_yaml=args.data,
        imgsz=args.imgsz,
        device=args.device,
        confidence=args.confidence,
        iou=args.iou,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    yaml_path = args.output.with_suffix(".yaml")
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print(f"Metrics saved to: {args.output}")
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()