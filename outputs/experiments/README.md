# Таблица экспериментов

Заполняется автоматически после `python src/train.py --config configs/experiments/<name>.yaml`
и вручную дополняется для отчёта.

| Модель | Архитектура | imgsz | epochs | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Latency (ms) | Вес (MB) | Вывод |
|--------|-------------|-------|--------|---------|--------------|-----------|--------|--------------|----------|-------|
| yolov8n_obb | YOLO OBB | 1024 | 50 | — | — | — | — | — | ~6.3 | Baseline |
| yolov8s_obb | YOLO OBB | 1024 | 50 | — | — | — | — | — | — | — |
| yolov8m_obb | YOLO OBB | 1024 | 50 | — | — | — | — | — | — | — |
| yolo11n_obb | YOLO11 OBB | 1024 | 50 | — | — | — | — | — | — | Edge |
| rtdetr_l | RT-DETR | 1024 | 50 | — | — | — | — | — | — | Transformer |

JSON-файлы с метриками: `outputs/experiments/<experiment_name>.json`