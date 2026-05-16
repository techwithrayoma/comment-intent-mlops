from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from src.core.config import get_settings

# ── import all models so Base.metadata is fully populated ────────────────────
from src.db.models.base import Base
from src.db.models.model import Model                        
from src.db.models.model_evaluation import ModelEvaluation  
from src.db.models.model_intent_label import ModelIntentLabel 
from src.db.models.model_evaluation import ModelEvaluation


# ── engine ────────────────────────────────────────────────────────────────────
engine = create_engine(
    get_settings().DATABASE_URL,
    echo=False,        
    pool_pre_ping=True # drops stale connections automatically
)

# ── session factory ───────────────────────────────────────────────────────────
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)

# ── dependency / context manager ─────────────────────────────────────────────
def get_db() -> Generator[Session, None, None]:
    """
    Use in Celery tasks:
        db = SessionLocal()
        try:
            ...
        finally:
            db.close()

    Or as a FastAPI dependency if you ever add sync endpoints:
        def my_route(db: Session = Depends(get_db)): ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()