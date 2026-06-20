# DOTADetector

Система детекции объектов на аэрофотоснимках (датасет DOTAv1) с обучением, сравнением архитектур и FastAPI-сервисом для инференса.

Соответствует заданию на практику: полный цикл CV-проекта — данные, обучение, метрики, деплой, история запусков.

## Структура проекта

```
Dotadetector/
├── configs/
│   ├── inference.yaml          # конфиг API-сервиса
│   └── experiments/          # 5 экспериментов для сравнения архитектур
├── DOTAv1.yaml                 # описание датасета для Ultralytics
├── src/
│   ├── prepare_dataset.py      # конвертация DOTA TXT → YOLO OBB
│   ├── train.py                # обучение по конфигу эксперимента
│   └── evaluate.py             # mAP@0.5, mAP@0.5:0.95, precision, recall
├── service/app/                # FastAPI + веб-интерфейс
├── models/models_info.yaml     # метаданные модели
├── weights/                    # веса для инференса
├── outputs/                    # метрики и таблица экспериментов
├── tests/                      # smoke-тесты API
├── Dockerfile
└── docker-compose.yml
```

## Установка

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Подготовка данных DOTAv1

1. Скачайте DOTAv1 и разложите по папкам:
   - `source_images/train`, `source_images/val`, `source_images/test`
   - `source_labels/train`, `source_labels/val`, `source_labels/test` (DOTA TXT)

2. Конвертируйте разметку в формат YOLO OBB:

```powershell
python src/prepare_dataset.py `
  --source-images "D:\datasets\DOTA\images" `
  --source-labels "D:\datasets\DOTA\labelTxt" `
  --output DOTAv1 `
  --dataset-yaml DOTAv1.yaml
```

3. Анализ баланса классов (для отчёта):

```powershell
python Histogram.py
```

## Обучение и сравнение архитектур (≥5 экспериментов)

| № | Конфиг | Архитектура |
|---|--------|-------------|
| 1 | `configs/experiments/yolov8n_obb.yaml` | YOLOv8n-OBB (baseline) |
| 2 | `configs/experiments/yolov8s_obb.yaml` | YOLOv8s-OBB |
| 3 | `configs/experiments/yolov8m_obb.yaml` | YOLOv8m-OBB |
| 4 | `configs/experiments/yolo11n_obb.yaml` | YOLO11n-OBB (edge) |
| 5 | `configs/experiments/rtdetr_l.yaml` | RT-DETR-L (transformer) |

Запуск одного эксперимента:

```powershell
python src/train.py --config configs/experiments/yolov8n_obb.yaml
```

Метрики сохраняются в `outputs/experiments/<name>.json`.

## Оценка качества (mAP, precision, recall, latency)

```powershell
python src/evaluate.py --weights weights/yolov8n-obb.pt --data DOTAv1.yaml
```

Результат: `outputs/metrics/latest_eval.json`

Обязательные метрики для отчёта по детекции:
- mAP@0.5
- mAP@0.5:0.95
- precision, recall
- latency (мс)
- анализ FP/FN на 5+ удачных и 5+ ошибочных примерах (скриншоты из `data/results/`)

## Gradio GUI (изображения + видео)

Рекомендуется для демонстрации и отчёта по заданию (раздел 5.1).

```powershell
pip install gradio opencv-python-headless
.\start_gradio.ps1
```

Откройте в браузере: **http://localhost:7860**

Вкладки:
- **Изображение** — загрузка, детекция, таблица объектов
- **Видео** — обработка MP4/AVI с аннотированным результатом
- **Модель и история** — классы, статистика, последние запуски

## Запуск API-сервиса

```powershell
.\start_server.ps1
```

или вручную:

```powershell
uvicorn service.app.main:app --host 0.0.0.0 --port 8000
```

**Важно:** `0.0.0.0` — это адрес привязки сервера, его нельзя открывать в браузере.
Открывайте: **http://localhost:8000** или **http://127.0.0.1:8000**

Терминал с сервером должен оставаться запущенным. Если видите `Shutting down` — сервер остановлен.

### Эндпоинты

| Метод | URL | Описание |
|-------|-----|----------|
| GET | `/health` | Статус, версия модели, список классов |
| POST | `/predict` | Детекция на одном изображении |
| POST | `/batch_predict` | До 10 изображений |
| GET | `/stats` | Статистика обработки |
| GET | `/history` | История запусков (SQLite) |
| GET | `/` | Веб-интерфейс |

Пример запроса:

```powershell
curl -X POST "http://localhost:8000/predict?confidence=0.25" `
  -F "file=@tests/test_data/P0723.png"
```

## Docker (одна команда)

```powershell
docker compose up --build
```

- FastAPI: http://localhost:8000
- Gradio GUI: http://localhost:7860

## Тесты

```powershell
pytest tests/ -v
```

Проверяют:
- старт сервиса и `/health`
- инференс на синтетическом и реальных DOTA-изображениях
- `/stats` и `/history`

## Для итогового отчёта

1. **Постановка задачи** — детекция объектов на снимках с дрона (DOTAv1)
2. **Данные** — DOTAv1, 15 классов, train/val/test, гистограммы (`Histogram.py`)
3. **Эксперименты** — таблица в `outputs/experiments/README.md`
4. **Метрики** — JSON из `src/evaluate.py`
5. **Деплой** — Docker + FastAPI + веб-UI + история в SQLite
6. **Демо** — скриншоты UI, `/docs`, примеры из `data/results/`