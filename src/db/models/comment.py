from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,  
    Index,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Comment(Base):
    """
    Raw comments table.

    One row per unique comment from any upstream source.
    Each comment belongs to one content record (video, post, etc.).
    """

    __tablename__ = "comments"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    # -------------------------------------------------------------------------
    # FOREIGN KEY TO CONTENTS TABLE
    # -------------------------------------------------------------------------
    # This links each comment to the content it came from.
    # Without this column, SQLAlchemy cannot build:
    # Content.comments <-> Comment.content
    # -------------------------------------------------------------------------
    content_id: Mapped[int] = mapped_column(
        ForeignKey("contents.id"),
        nullable=False,
        index=True,
    )


    author: Mapped[str | None] = mapped_column(
            String(255),
            nullable=True,
        )


    comment_text: Mapped[str] = mapped_column(Text, nullable=False)

    published_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    source: Mapped[str | None] = mapped_column(String(64), nullable=True)
    language: Mapped[str | None] = mapped_column(String(8), nullable=True)

    is_spam: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_comments_published_at", "published_at"),
        Index("ix_comments_is_spam", "is_spam"),
    )

    def __repr__(self) -> str:
        return f"<Comment id={self.id} content_id={self.content_id}>"