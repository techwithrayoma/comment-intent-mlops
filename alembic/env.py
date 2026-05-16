import sys
from pathlib import Path
from logging.config import fileConfig

from sqlalchemy import create_engine, pool
from alembic import context

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import get_settings
from src.db.models.base import Base
from src.db.models.model import Model
from src.db.models.model_evaluation import ModelEvaluation
from src.db.models.model_intent_label import ModelIntentLabel 
from src.db.models.model_evaluation import ModelEvaluation

# =============================================================================
# TABLES OWNED BY THIS REPOSITORY
# =============================================================================
# This training repository is responsible only for these tables.
# Alembic will ignore all other tables in the database
# (for example: comments and contents from the scraper repository).
# This prevents Alembic from generating DROP TABLE statements
# for tables that belong to another repository.
# =============================================================================
OWNED_TABLES = {
    "models",
    "model_evaluations",
    "model_intent_labels",
    "model_evaluations",
}


def include_object(object_, name, type_, reflected, compare_to):
    """
    Control which database objects Alembic should include.

    If the object is a table, include it only if it belongs to this repo.
    Ignore all tables owned by other repositories.
    """
    if type_ == "table":
        return name in OWNED_TABLES
    return True


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        include_object=include_object,              
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table="alembic_version_training",
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(
        settings.DATABASE_URL,
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,       
            version_table="alembic_version_training",
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()