from datetime import datetime, timezone
from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class Model(Base):
    """
    Represents a trained or fine-tuned model version.
    Created by TrainPipeline when a training run completes.
    """

    __tablename__ = "models"

    id                     : Mapped[int]        = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name             : Mapped[str]        = mapped_column(String(128), nullable=False)
    model_version          : Mapped[str]        = mapped_column(String(64),  nullable=False)
    mlflow_run_id          : Mapped[str | None] = mapped_column(String(256), nullable=True)
    total_training_cost_usd: Mapped[float|None] = mapped_column(Float,       nullable=True)
    gpu_type               : Mapped[str | None] = mapped_column(String(64),  nullable=True)
    gpu_hours              : Mapped[float|None] = mapped_column(Float,       nullable=True)
    is_active              : Mapped[bool]       = mapped_column(Boolean,     nullable=False, default=False)
    created_at             : Mapped[datetime]   = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # ── relationships ─────────────────────────────────────────────────────────
    evaluations = relationship("ModelEvaluation", back_populates="model", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Model {self.model_name}@{self.model_version} active={self.is_active}>"