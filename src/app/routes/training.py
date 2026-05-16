from fastapi import APIRouter

from src.app.routes.schemas.training import (
    CleanRequest,
    IngestionRequest,
    LabelingRequest,
    TaskResponse,
    TrainingAssetsRequest,
    TrainingRequest,
    TransformationRequest,
)
from src.tasks.training_tasks import (
    task_data_clean,
    task_data_ingestion,
    task_data_labeling,
    task_data_transformation,
    task_prepare_training_assets,
    task_run_model_training
)

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.post("/ingest", response_model=TaskResponse, summary="Step 1: Ingest training data")
async def trigger_ingestion(req: IngestionRequest):
    task = task_data_ingestion.delay(**req.model_dump())
    return TaskResponse(
        status="accepted",
        task_id=task.id,
        message=f"Ingestion queued for {req.model_name}/{req.version}",
    )


@router.post("/clean", response_model=TaskResponse, summary="Step 2: Clean raw data")
async def trigger_clean(req: CleanRequest):
    task = task_data_clean.delay(**req.model_dump())
    return TaskResponse(
        status="accepted",
        task_id=task.id,
        message=f"Clean queued for {req.model_name}/{req.version}",
    )


@router.post("/label", response_model=TaskResponse, summary="Step 3: LLM label clean data")
async def trigger_labeling(req: LabelingRequest):
    task = task_data_labeling.delay(**req.model_dump())
    return TaskResponse(
        status="accepted",
        task_id=task.id,
        message=f"Labeling queued for {req.model_name}/{req.version}",
    )


@router.post("/transform", response_model=TaskResponse, summary="Step 4: Transform for fine-tuning")
async def trigger_transformation(req: TransformationRequest):
    task = task_data_transformation.delay(**req.model_dump())
    return TaskResponse(
        status="accepted",
        task_id=task.id,
        message=f"Transformation queued for {req.model_name}/{req.version}",
    )


@router.post("/prepare-assets", response_model=TaskResponse, summary="Step 5: Build JSONL + config")
async def trigger_prepare_assets(req: TrainingAssetsRequest):
    # Extract overrides as a plain dict, dropping None values
    # None = "use YAML default for this key"
    overrides = req.overrides.model_dump(exclude_none=True) or None

    task = task_prepare_training_assets.delay(
        model_name=req.model_name,
        version=req.version,
        store_s3=req.store_s3,
        store_local=req.store_local,
        overrides=overrides,
    )
    return TaskResponse(
        status="accepted",
        task_id=task.id,
        message=f"Asset preparation queued for {req.model_name}/{req.version} | overrides={overrides}",
    )


@router.post("/train", response_model=TaskResponse, summary="Step 6: GPU fine-tune + evaluate")
async def trigger_training(req: TrainingRequest):
    task = task_run_model_training.delay(**req.model_dump())
    return TaskResponse(
        status="accepted",
        task_id=task.id,
        message=f"GPU training queued for {req.model_name}/{req.version}",
    )