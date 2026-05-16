from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import text
from src.db.models.comment import Comment

class CommentRepository:

    def __init__(self, db_session: Session):
        self.db = db_session


    def get_training_comments(
        self,
        min_char_length: int = 20,
        max_char_length: int = 2000,
        max_rows:        int = 50_000,
    ) -> list[dict]:
        
        query = (
            self.db.query(
                Comment.id.label("comment_id"),
                Comment.comment_text,
                Comment.published_at,
                Comment.source,
                Comment.language,
            )
            .filter(
                Comment.is_spam == False,
                text("char_length(comment_text) >= :min_len"),
                text("char_length(comment_text) <= :max_len"),
            )
            .params(min_len=min_char_length, max_len=max_char_length)
            .order_by(Comment.published_at.asc())
            .limit(max_rows)
        )

        return [row._asdict() for row in query.all()]