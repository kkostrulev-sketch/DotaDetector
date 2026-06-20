from __future__ import annotations
import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import base64
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from service.app.config import load_settings
from service.app.detector import DDetector
from service.app.database import HistoryStore
from service.app.schemas import HealthResponse, HistoryItem, PredictResponse, StatsResponse

settings = load_settings()
detector: DDetector | None = None
history_store: HistoryStore | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global detector, history_store
    detector = DDetector(settings)
    history_store = HistoryStore(settings.history_db, settings.results_dir)
    yield


app = FastAPI(
    title="Drone Object Detection Service",
    description="Сервис детекции объектов на снимках с дрона (YOLOv8)",
    version=settings.model_version,
    lifespan=lifespan,
)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    classes = detector.get_class_names_list() if detector is not None else settings.classes

    return HealthResponse(
        status="ok",
        model_loaded=detector is not None,
        model_version=settings.model_version,
        architecture=settings.architecture,
        classes=classes,
    )


@app.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    if history_store is None:
        raise HTTPException(status_code=503, detail="History store is not ready")
    return StatsResponse(**history_store.get_stats())


@app.get("/history", response_model=list[HistoryItem])
async def history(limit: int = Query(default=20, ge=1, le=100)) -> list[HistoryItem]:
    if history_store is None:
        raise HTTPException(status_code=503, detail="History store is not ready")
    return [HistoryItem(**item) for item in history_store.list_predictions(limit=limit)]


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    confidence: float = Query(default=settings.confidence_threshold, ge=0.05, le=0.95),
) -> PredictResponse:
    if detector is None or history_store is None:
        raise HTTPException(status_code=503, detail="Detector is not ready")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")

    image_bytes = await file.read()
    max_size = settings.max_upload_size_mb * 1024 * 1024
    if len(image_bytes) > max_size:
        raise HTTPException(status_code=400, detail=f"Размер файла превышает {settings.max_upload_size_mb} МБ")

    filename = file.filename or "upload.jpg"

    try:
        result = detector.predict(image_bytes, confidence=confidence)
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
        record_id = history_store.add_prediction(
            filename=filename,
            model_version=settings.model_version,
            detections=detections_payload,
            latency_ms=result.latency_ms,
            confidence_threshold=confidence,
            annotated_image_bytes=annotated_bytes,
        )
    except Exception as exc:
        history_store.add_prediction(
            filename=filename,
            model_version=settings.model_version,
            detections=[],
            latency_ms=0.0,
            confidence_threshold=confidence,
            error=str(exc),
        )
        raise HTTPException(status_code=500, detail=f"Ошибка инференса: {exc}") from exc

    return PredictResponse(
        id=record_id,
        filename=filename,
        model_version=settings.model_version,
        architecture=settings.architecture,
        detections=detections_payload,
        detections_count=len(detections_payload),
        latency_ms=round(result.latency_ms, 2),
        image_width=result.image_width,
        image_height=result.image_height,
        annotated_image_base64=result.annotated_image_base64,
    )


@app.post("/batch_predict")
async def batch_predict(
    files: list[UploadFile] = File(...),
    confidence: float = Query(default=settings.confidence_threshold, ge=0.05, le=0.95),
) -> list[PredictResponse]:
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Максимум 10 изображений за один запрос")

    responses: list[PredictResponse] = []
    for upload in files:
        responses.append(await predict(file=upload, confidence=confidence))
    return responses