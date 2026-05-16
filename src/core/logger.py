import logging

# -------------------------------
# Task ID filter
# -------------------------------
class TaskIdFilter(logging.Filter):
    """Add Celery task ID to log records if available."""
    def __init__(self, task_id="no-task"):
        super().__init__()
        self.task_id = task_id

    def set_task_id(self, task_id: str):
        self.task_id = task_id

    def filter(self, record):
        record.task_id = self.task_id
        return True


# -------------------------------
# Pipeline logger
# -------------------------------
pipeline_logger = logging.getLogger("PipelineLogger")
pipeline_logger.setLevel(logging.INFO)

task_filter = TaskIdFilter()
pipeline_logger.addFilter(task_filter)

if not pipeline_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(task_id)s] %(message)s"
    )
    handler.setFormatter(formatter)
    pipeline_logger.addHandler(handler)

# Prevent Celery root logger from duplicating logs
pipeline_logger.propagate = False

# -------------------------------
# Silence noisy third-party loggers
# -------------------------------
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
