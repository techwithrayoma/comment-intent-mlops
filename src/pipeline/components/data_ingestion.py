from datetime import datetime
import pandas as pd
from sqlalchemy.orm import Session

from src.repository.comment_repository import CommentRepository


class IngestData:

    def __init__(self, db: Session, model_name: str, version: str):
        self.repo       = CommentRepository(db)
        self.model_name = model_name
        self.version    = version
        self._df: pd.DataFrame | None = None

    def fetch(
        self,
        min_char_length: int = 20,
        max_char_length: int = 2000,
        max_rows:        int = 50_000,
    ) -> pd.DataFrame:
        rows = self.repo.get_training_comments(
            min_char_length = min_char_length,
            max_char_length = max_char_length,
            max_rows        = max_rows,
        )
        self._df = pd.DataFrame(rows) if rows else pd.DataFrame()
        return self._df
    

    