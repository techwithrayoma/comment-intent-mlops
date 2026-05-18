from datetime import datetime
import uuid
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import (
    BigInteger,
    DateTime,
    Index,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Comment(Base):
    __tablename__ = "comments"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    comment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        default=uuid.uuid4,
        unique=True,
        index=True,
        nullable=False,
    )

    comment_text: Mapped[str] = mapped_column(Text, nullable=False)

    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_comments_published_at", "published_at"),
    )

    def __repr__(self) -> str:
        return f"<Comment id={self.id} content_id={self.content_id}>"