"""Train detection models on DOTAv1 using experiment configs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "configs" / "experiments"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "experiments"


def load_experiment_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def train_experiment(config_path: Path) -> dict:
    from ultralytics import YOLO

    config = load_experiment_config(config_path)
    experiment_name = config.get("name", config_path.stem)
    train_cfg = config["train"]

    model = YOLO(train_cfg["model"])
    started_at = datetime.now(timezone.utc).isoformat()

    results = model.train(
        data=str((PROJECT_ROOT / train_cfg["data"]).resolve()),
        epochs=int(train_cfg.get("epochs", 50)),
        imgsz=int(train_cfg.get("imgsz", 1024)),
        batch=int(train_cfg.get("batch", 8)),
        device=train_cfg.get("device", "auto"),
        project=str((PROJECT_ROOT / train_cfg.get("project", "runs/train")).resolve()),
        name=experiment_name,
        patience=int(train_cfg.get("patience", 20)),
        optimizer=train_cfg.get("optimizer", "auto"),
        lr0=float(train_cfg.get("lr0", 0.01)),
        verbose=True,
    )

    metrics = getattr(results, "results_dict", {}) or {}
    save_dir = Path(getattr(results, "save_dir", PROJECT_ROOT / "runs" / "train" / experiment_name))

    summary = {
        "experiment": experiment_name,
        "config": str(config_path),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "weights": str(save_dir / "weights" / "best.pt"),
        "save_dir": str(save_dir),
        "metrics": {
            "mAP50": float(metrics.get("metrics/mAP50(B)", 0.0)),
            "mAP50-95": float(metrics.get("metrics/mAP50-95(B)", 0.0)),
            "precision": float(metrics.get("metrics/precision(B)", 0.0)),
            "recall": float(metrics.get("metrics/recall(B)", 0.0)),
        },
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUTS_DIR / f"{experiment_name}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Experiment summary saved to: {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a DOTA detection training experiment")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment yaml in configs/experiments/",
    )
    args = parser.parse_args()
    train_experiment(args.config.resolve())


if __name__ == "__main__":
    main()