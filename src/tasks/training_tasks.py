import time

from src.app.celery_app import celery_app
from src.core.config import get_settings
from src.core.logger import TaskIdFilter, pipeline_logger
from src.db.database import SessionLocal
from src.pipeline.pipeline_enum import PipelineEnum
from src.pipeline.training_pipeline import TrainPipeline

logger      = pipeline_logger
task_filter = TaskIdFilter()
pipeline_logger.addFilter(task_filter)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_pipeline(
    version:     str,
    model_name:  str,
    task_id:     str,
    store_s3:    bool,
    store_local: bool,
) -> TrainPipeline:
    return TrainPipeline(
        config=get_settings(),
        version=version,
        model_name=model_name,
        store_s3=store_s3,
        store_local=store_local,
        task_id=task_id,
    )


def _get_queue(task) -> str:
    return task.request.delivery_info.get("routing_key", "unknown")


def _timed(label: str, fn, *args, **kwargs):
    """Run fn(*args, **kwargs), log elapsed time, return result."""
    t = time.perf_counter()
    result = fn(*args, **kwargs)
    logger.info(f"[{label}] done in {time.perf_counter() - t:.2f}s")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — INGESTION
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(bind=True, queue="cpu", name="tasks.training_tasks.task_data_ingestion")
def task_data_ingestion(
    self,
    model_name:      str,
    version:         str,
    min_char_length: int  = 20,
    max_char_length: int  = 2000,
    max_rows:        int  = 50_000,
    store_s3:        bool = True,
    store_local:     bool = False,
):
    task_id = self.request.id
    task_filter.set_task_id(task_id)
    db = None

    try:
        logger.info(
            f"[{PipelineEnum.INGESTION}] started | "
            f"version={version} | queue={_get_queue(self)}"
        )
        db       = SessionLocal()
        pipeline = _build_pipeline(version, model_name, task_id, store_s3, store_local)

        _timed(PipelineEnum.STORAGE,    pipeline.create_folders)
        _timed(PipelineEnum.INGESTION,  pipeline.data_ingestion,
               db,
               min_char_length=min_char_length,
               max_char_length=max_char_length,
               max_rows=max_rows)

        return {"status": "done", "step": "ingestion", "version": version}

    except Exception as e:
        logger.error(f"[{PipelineEnum.INGESTION}] failed: {e}", exc_info=True)
        raise
    finally:
        if db:
            db.close()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — CLEAN
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(bind=True, queue="cpu", name="tasks.training_tasks.task_data_clean")
def task_data_clean(
    self,
    model_name:  str,
    version:     str,
    store_s3:    bool = True,
    store_local: bool = False,
):
    task_id = self.request.id
    task_filter.set_task_id(task_id)

    try:
        logger.info(
            f"[{PipelineEnum.CLEAN}] started | "
            f"version={version} | queue={_get_queue(self)}"
        )
        pipeline = _build_pipeline(version, model_name, task_id, store_s3, store_local)
        _timed(PipelineEnum.CLEAN, pipeline.data_clean)

        return {"status": "done", "step": "clean", "version": version}

    except Exception as e:
        logger.error(f"[{PipelineEnum.CLEAN}] failed: {e}", exc_info=True)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — LABELING
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(bind=True, queue="cpu", name="tasks.training_tasks.task_data_labeling")
def task_data_labeling(
    self,
    model_name:  str,
    version:     str,
    store_s3:    bool = True,
    store_local: bool = False,
):
    task_id = self.request.id
    task_filter.set_task_id(task_id)
    db = None

    try:
        logger.info(
            f"[{PipelineEnum.LABELING}] started | "
            f"version={version} | queue={_get_queue(self)}"
        )
        db       = SessionLocal()
        pipeline = _build_pipeline(version, model_name, task_id, store_s3, store_local)
        _timed(PipelineEnum.LABELING, pipeline.data_labeling, db)

        return {"status": "done", "step": "labeling", "version": version}

    except Exception as e:
        logger.error(f"[{PipelineEnum.LABELING}] failed: {e}", exc_info=True)
        raise
    finally:
        if db:
            db.close()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — TRANSFORMATION
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(bind=True, queue="cpu", name="tasks.training_tasks.task_data_transformation")
def task_data_transformation(
    self,
    model_name:  str,
    version:     str,
    store_s3:    bool = True,
    store_local: bool = False,
):
    task_id = self.request.id
    task_filter.set_task_id(task_id)

    try:
        logger.info(
            f"[{PipelineEnum.TRANSFORMATION}] started | "
            f"version={version} | queue={_get_queue(self)}"
        )
        pipeline = _build_pipeline(version, model_name, task_id, store_s3, store_local)
        _timed(PipelineEnum.TRANSFORMATION, pipeline.data_transformation)

        return {"status": "done", "step": "transformation", "version": version}

    except Exception as e:
        logger.error(f"[{PipelineEnum.TRANSFORMATION}] failed: {e}", exc_info=True)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — PREPARE TRAINING ASSETS
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(bind=True, queue="cpu", name="tasks.training_tasks.task_prepare_training_assets")
def task_prepare_training_assets(
    self,
    model_name:  str,
    version:     str,
    overrides:   dict | None = None,
    store_s3:    bool = True,
    store_local: bool = True,
):
    task_id = self.request.id
    task_filter.set_task_id(task_id)

    try:
        logger.info(
            f"[{PipelineEnum.PRE_TRAINING}] prepare assets started | "
            f"version={version} | overrides={overrides} | queue={_get_queue(self)}"
        )
        pipeline = _build_pipeline(version, model_name, task_id, store_s3, store_local)
        _timed(PipelineEnum.PRE_TRAINING, pipeline.prepare_training_assets, overrides)

        return {"status": "done", "step": "prepare_assets", "version": version}

    except Exception as e:
        logger.error(f"[{PipelineEnum.PRE_TRAINING}] prepare assets failed: {e}", exc_info=True)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — MODEL TRAINING + EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(bind=True, queue="gpu", name="tasks.training_tasks.task_run_model_training")
def task_run_model_training(
    self,
    model_name:  str,
    version:     str,
    store_s3:    bool = True,
    store_local: bool = False,
):
    task_id = self.request.id
    task_filter.set_task_id(task_id)
    db = None

    try:
        logger.info(
            f"[{PipelineEnum.TRAINING}] GPU training started | "
            f"version={version} | queue={_get_queue(self)}"
        )
        db       = SessionLocal()
        pipeline = _build_pipeline(version, model_name, task_id, store_s3, store_local)
        _timed(PipelineEnum.TRAINING, pipeline.run_model_training, db)

        return {"status": "done", "step": "model_training", "version": version}

    except Exception as e:
        logger.error(f"[{PipelineEnum.TRAINING}] GPU training failed: {e}", exc_info=True)
        raise
    finally:
        if db:
            db.close()