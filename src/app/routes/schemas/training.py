from typing import Optional
from pydantic import BaseModel, Field


class _BaseRequest(BaseModel):
    model_name:  str  = "comment-intent"
    version:     str
    store_s3:    bool = True
    store_local: bool = True


class IngestionRequest(_BaseRequest):
    min_char_length: int = Field(20,     ge=1)
    max_char_length: int = Field(2000,   le=10_000)
    max_rows:        int = Field(50_000, ge=1)


class CleanRequest(_BaseRequest):
    pass


class LabelingRequest(_BaseRequest):
    pass


class TransformationRequest(_BaseRequest):
    pass


class TrainingOverrides(BaseModel):
    """
    Optional per-run hyperparameter overrides.
    Only set what you want to change — everything else comes from the YAML.
    """
    learning_rate:               Optional[float] = None
    num_train_epochs:            Optional[float] = None
    per_device_train_batch_size: Optional[int]   = None
    gradient_accumulation_steps: Optional[int]   = None
    lora_rank:                   Optional[int]   = None
    cutoff_len:                  Optional[int]   = None
    warmup_ratio:                Optional[float] = None


class TrainingAssetsRequest(_BaseRequest):
    overrides: TrainingOverrides = Field(default_factory=TrainingOverrides)


class TrainingRequest(_BaseRequest):
    pass


class TaskResponse(BaseModel):
    status:   str
    task_id:  str
    message:  str