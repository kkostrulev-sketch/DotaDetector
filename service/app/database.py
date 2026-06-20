from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional  # <--- Добавлен импорт Optional

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    detections_count: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    detections_json: Mapped[str] = mapped_column(Text, nullable=False)

    # <--- Заменено str | None на Optional[str]
    result_image_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class HistoryStore:
    def __init__(self, db_path: Path, results_dir: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = results_dir
        self.engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
        self.session_factory = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
        Base.metadata.create_all(self.engine)

    def add_prediction(
            self,
            filename: str,
            model_version: str,
            detections: list[dict],
            latency_ms: float,
            confidence_threshold: float,
            annotated_image_bytes: Optional[bytes] = None,  # <--- Заменено bytes | None
            error: Optional[str] = None,  # <--- Заменено str | None
    ) -> int:
        result_path: Optional[str] = None  # <--- Заменено str | None

        if annotated_image_bytes:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            safe_name = Path(filename).stem[:80] or "image"
            result_path = str(self.results_dir / f"{timestamp}_{safe_name}.jpg")
            Path(result_path).write_bytes(annotated_image_bytes)

        record = PredictionHistory(
            created_at=datetime.now(timezone.utc),
            filename=filename,
            model_version=model_version,
            detections_count=len(detections),
            latency_ms=latency_ms,
            confidence_threshold=confidence_threshold,
            detections_json=json.dumps(detections, ensure_ascii=False),
            result_image_path=result_path,
            error=error,
        )

        with Session(self.engine) as session:
            session.add(record)
            session.commit()
            session.refresh(record)
            return int(record.id)

    def list_predictions(self, limit: int = 20) -> list[dict]:
        with Session(self.engine) as session:
            rows = session.scalars(
                select(PredictionHistory).order_by(PredictionHistory.id.desc()).limit(limit)
            ).all()
            return [self._to_dict(row) for row in rows]

    def get_stats(self) -> dict:
        with Session(self.engine) as session:
            total = session.scalar(select(func.count()).select_from(PredictionHistory)) or 0
            avg_latency = session.scalar(select(func.avg(PredictionHistory.latency_ms))) or 0.0
            avg_detections = session.scalar(select(func.avg(PredictionHistory.detections_count))) or 0.0
            errors = session.scalar(
                select(func.count()).select_from(PredictionHistory).where(PredictionHistory.error.is_not(None))
            ) or 0
            return {
                "total_predictions": int(total),
                "average_latency_ms": round(float(avg_latency), 2),
                "average_detections": round(float(avg_detections), 2),
                "error_count": int(errors),
            }

    @staticmethod
    def _to_dict(row: PredictionHistory) -> dict:
        return {
            "id": row.id,
            "created_at": row.created_at.isoformat(),
            "filename": row.filename,
            "model_version": row.model_version,
            "detections_count": row.detections_count,
            "latency_ms": round(row.latency_ms, 2),
            "confidence_threshold": row.confidence_threshold,
            "detections": json.loads(row.detections_json),
            "result_image_path": row.result_image_path,
            "error": row.error,
        }