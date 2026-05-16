from sqlalchemy.orm import Session
from src.db.models.model_evaluation import ModelEvaluation


class ModelEvaluationRepository:

    def __init__(self, db: Session):
        self.db = db

    def create_evaluation(
        self,
        model_id:                int,
        accuracy:                float,
        f1_score:                float,
        evaluation_dataset_hash: str | None = None,
    ) -> int:
        record = ModelEvaluation(
            model_id=model_id,
            accuracy=accuracy,
            f1_score=f1_score,
            evaluation_dataset_hash=evaluation_dataset_hash,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record.id