from pathlib import Path
from pydantic_settings import BaseSettings



class Settings(BaseSettings):
    
    # Database credentials
    # =========================
    DATABASE_URL: str

    
    # Cloud Storage Configuration
    # =========================
    STORAGE_PROVIDER: str

    AWS_SECRET_ACCESS_KEY: str
    AWS_ACCESS_KEY_ID: str
    AWS_REGION: str
    S3_BUCKET_NAME: str


    # LLM Configuration
    # =========================
    LLM_PROVIDER: str
    
    OPENAI_API_KEY: str
    OPENAI_MODEL_ID: str
    OPENAI_API_BASE_URL: str
    OPENAI_MODEL_ID: str
    OPENAI_MAX_INPUT_CHARS: int
    OPENAI_MAX_OUTPUT_TOKENS: int
    OPENAI_TEMPERATURE: float
    OPENAI_INPUT_PRICING: float
    OPENAI_OUTPUT_PRICING: float

    # Celery Configuration
    # =========================
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    
    CELERY_TASK_SERIALIZER: str
    CELERY_TASK_TIME_LIMIT: int
    CELERY_TASK_ACKS_LATE: bool
    CELERY_WORKER_CONCURRENCY_CPU: int
    CELERY_WORKER_CONCURRENCY_GPU: int

    mlflow_tracking_uri: str
    MODEL_NAME: str
    
    class Config:
        env_file = str(Path(__file__).resolve().parents[1] / ".env")


def get_settings() -> Settings:
    return Settings()
