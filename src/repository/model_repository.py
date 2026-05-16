from sqlalchemy.orm import Session
from src.db.models.model import Model


class ModelRepository:

    def __init__(self, db: Session):
        self.db = db

    def get_by_version(self, model_name: str, model_version: str) -> Model | None:
        return (
            self.db.query(Model)
            .filter(
                Model.model_name    == model_name,
                Model.model_version == model_version,
            )
            .first()
        )

    def create_model_run(
        self,
        model_name:              str,
        model_version:           str,
        mlflow_run_id:           str | None = None,
        gpu_type:                str | None = None,
        gpu_hours:               float | None = None,
        total_training_cost_usd: float | None = None,
    ) -> int:
        """Insert a new model record and return its id."""
        record = Model(
            model_name=model_name,
            model_version=model_version,
            mlflow_run_id=mlflow_run_id,
            gpu_type=gpu_type,
            gpu_hours=gpu_hours,
            total_training_cost_usd=total_training_cost_usd,
            is_active=False,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record.id