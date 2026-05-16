from datetime import datetime, timezone
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class ModelEvaluation(Base):
    """
    Stores benchmark evaluation results for a model version.
    Created by ModelEvaluation step after training completes.
    """

    __tablename__ = "model_evaluations"

    id                      : Mapped[int]        = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id                : Mapped[int]        = mapped_column(
        Integer, ForeignKey("models.id", ondelete="CASCADE"), nullable=False
    )
    accuracy                : Mapped[float]      = mapped_column(Float,        nullable=False)
    f1_score                : Mapped[float]      = mapped_column(Float,        nullable=False)
    evaluation_dataset_hash : Mapped[str | None] = mapped_column(String(256),  nullable=True)
    created_at              : Mapped[datetime]   = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # ── relationships ─────────────────────────────────────────────────────────
    model = relationship("Model", back_populates="evaluations")

    def __repr__(self) -> str:
        return f"<ModelEvaluation model_id={self.model_id} acc={self.accuracy:.3f} f1={self.f1_score:.3f}>"