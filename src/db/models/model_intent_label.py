from datetime import datetime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
import uuid
from sqlalchemy import (
    BigInteger,
    DateTime,
    Index,
    String,
    Float,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class ModelIntentLabel(Base):

    __tablename__ = "model_intent_labels"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    comment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        index=True,
        nullable=False,
    )

    model_name: Mapped[str] = mapped_column(String(100), nullable=False)

    version: Mapped[str] = mapped_column(String(50), nullable=False)

    intent: Mapped[str] = mapped_column(String(50),nullable=False, index=True)
    # Complaint | Question | Suggestion | Praise | Statement

    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    label_source: Mapped[str | None] = mapped_column(String(50), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "comment_id",
            "model_name",
            "version",
            name="uq_comment_model_version",
        ),
        Index("ix_labels_model_version", "model_name", "version"),
        Index("ix_labels_intent", "intent"),
    )

    def __repr__(self) -> str:
        return (
            f"<ModelIntentLabel comment_id={self.comment_id} "
            f"model={self.model_name} version={self.version} "
            f"intent={self.intent}>"
        )
    

