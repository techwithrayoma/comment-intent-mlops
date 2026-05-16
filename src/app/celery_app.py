from celery import Celery

from src.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "comment_intent",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "tasks.training_tasks",
    ],
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Reliability
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_ignore_result=False,
    result_expires=86400,  # 24 hours

    # Timeouts
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,   # hard kill
    task_soft_time_limit=settings.CELERY_TASK_TIME_LIMIT - 60,  # graceful

    # Logging — don't let Celery hijack root logger
    worker_hijack_root_logger=False,

    # Route tasks to correct queues
    task_routes={
        "tasks.training_tasks.task_data_ingestion": {"queue": "cpu"},
        "tasks.training_tasks.task_data_clean": {"queue": "cpu"},
        "tasks.training_tasks.task_data_labeling": {"queue": "cpu"},
        "tasks.training_tasks.task_data_transformation": {"queue": "cpu"},
        "tasks.training_tasks.task_prepare_training_assets": {"queue": "cpu"},
        "tasks.training_tasks.task_run_model_training": {"queue": "gpu"},
    },
)