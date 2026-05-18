import hashlib
import os
import boto3
import pandas as pd

from src.pipeline.components.data_clean import DataClean
from src.pipeline.components.data_ingestion import IngestData
from src.pipeline.components.data_labeling import DataLabeling
from src.pipeline.components.data_transformation import DataTransformation
from src.repository.model_repository import ModelRepository
from src.repository.model_evaluation_repository import ModelEvaluationRepository
from src.pipeline.components.model_evaluation import ModelEvaluation
from src.pipeline.components.model_training import ModelTraining
from src.core.load_yaml import load_training_config
from src.store.llm.llm_provider_factory import LLMProviderFactory
from src.store.storage_manager import LocalBackend, S3Backend, StorageManager
from .pipeline_enum import PipelineEnum
from src.repository.model_intent_label_repository import ModelIntentLabelRepository
from src.core.logger import pipeline_logger


class TrainPipeline:
    """End-to-end ML training pipeline with step-level caching."""

    def __init__(
        self,
        config,
        version:     str,
        store_s3:    bool = True,
        store_local: bool = True,
        model_name:  str  = "comment-intent",
        task_id:     str  = "no-task",
    ):
        self.version    = version
        self.model_name = model_name
        self.config     = config

        self.logger = pipeline_logger
        self.logger.addFilter(
            type(
                "_TaskFilter",
                (),
                {"filter": lambda self, r: setattr(r, "task_id", task_id) or True},
            )()
        )
        self.logger.info(
            f"[{PipelineEnum.PIPELINE}] Initializing "
            f"model='{self.model_name}' version='{self.version}'"
        )

        llm_factory       = LLMProviderFactory(config)
        self.llm_provider = llm_factory.create(config.LLM_PROVIDER)

        local_backend = LocalBackend(base_path="/app/src/assets") if store_local else None
        s3_backend    = None
        if store_s3:
            s3_client = boto3.client(
                "s3",
                aws_access_key_id     = config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key = config.AWS_SECRET_ACCESS_KEY,
                region_name           = config.AWS_REGION,
            )
            s3_backend = S3Backend(s3_client=s3_client, bucket=config.S3_BUCKET_NAME)

        self.storage = StorageManager(
            project=self.model_name,
            version=self.version,
            local_backend=local_backend,
            s3_backend=s3_backend,
        )
        self.logger.info(f"[{PipelineEnum.PIPELINE}] TrainPipeline ready")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _cached(self, stage: str, filename: str) -> bool:
        if self.storage.exists(stage, filename):
            self.logger.info(
                f"[{PipelineEnum.PIPELINE}] [CACHE] {stage}/{filename} exists — skipping."
            )
            return True
        return False

    def _hash_df(self, df: pd.DataFrame) -> str:
        return hashlib.sha256(
            pd.util.hash_pandas_object(df, index=True).values
        ).hexdigest()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 0 — FOLDERS
    # ─────────────────────────────────────────────────────────────────────────

    def create_folders(self):
        if self.storage.stage_exists("training") or self.storage.stage_exists("raw_data"):
            self.logger.info(f"[{PipelineEnum.STORAGE}] [CACHE] Folders exist — skipping.")
            return

        self.logger.info(f"[{PipelineEnum.STORAGE}] Creating folder structure...")
        self.storage.create_folder_structure()
        self.storage.copy_benchmark_from_root()
        self.logger.info(f"[{PipelineEnum.STORAGE}] Folders ready.")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — DATA INGESTION
    # ─────────────────────────────────────────────────────────────────────────

    def data_ingestion(
        self,
        db,
        min_char_length: int = 20,
        max_char_length: int = 2000,
        max_rows:        int = 50_000,
    ) -> None:
        if self._cached("raw_data", "raw_data.csv"):
            return

        self.logger.info(f"[{PipelineEnum.INGESTION}] Fetching all labeled comments...")

        ingester = IngestData(db=db, model_name=self.model_name, version=self.version)
        df = ingester.fetch(
            min_char_length=min_char_length,
            max_char_length=max_char_length,
            max_rows=max_rows,
        )

        if df.empty:
            raise ValueError("No comments found. Check the database has data.")

        self.storage.save_df(df=df, stage="raw_data", filename="raw_data.csv")
        self.logger.info(f"[{PipelineEnum.INGESTION}] Saved {len(df)} rows.")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — DATA CLEAN
    # ─────────────────────────────────────────────────────────────────────────

    def data_clean(self) -> None:
        if self._cached("processed_data", "clean_data.csv"):
            return

        self.logger.info(f"[{PipelineEnum.CLEAN}] Cleaning data...")
        df       = self.storage.load_df(stage="raw_data", filename="raw_data.csv")
        clean_df = DataClean(ingested_data=df).clean_training_data()

        if clean_df.empty:
            raise ValueError("DataClean produced an empty DataFrame.")

        self.storage.save_df(df=clean_df, stage="processed_data", filename="clean_data.csv")
        self.logger.info(f"[{PipelineEnum.CLEAN}] Done — {len(clean_df)} rows saved.")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — DATA LABELING
    # ─────────────────────────────────────────────────────────────────────────

    def data_labeling(self, db) -> None:

        label_repo = ModelIntentLabelRepository(db)

        # ── LEVEL 1: fully cached ─────────────────────────────────────────────
        if self._cached("processed_data", "label_data.csv"):

            # Check if DB write from a prior run was also completed.
            # We do this by reading the saved CSV and checking if the DB
            # already has records for these comment_ids + model + version.
            self.logger.info(
                f"[{PipelineEnum.LABELING}] CSV cache hit — "
                "checking DB for missing records..."
            )
            try:
                labeled_df    = self.storage.load_df(
                    stage="processed_data", filename="label_data.csv"
                )

                comment_ids   = labeled_df["comment_id"].dropna().astype(str).tolist() \
                                if "comment_id" in labeled_df.columns else []

                self.logger.info(
                    f"[{PipelineEnum.LABELING}] comment_ids found in CSV: {len(comment_ids)}"
                )
                
                existing      = label_repo.get_labels_for_comments(
                    comment_ids=comment_ids,
                    model_name=self.model_name,
                    version=self.version,
                )
                missing_ids   = set(comment_ids) - set(existing.keys())

                if not missing_ids:
                    self.logger.info(
                        f"[{PipelineEnum.LABELING}] [CACHE] "
                        "CSV + DB both complete — skipping."
                    )
                    return

                # DB write failed on prior run — recover by saving missing rows
                self.logger.warning(
                    f"[{PipelineEnum.LABELING}] CSV exists but "
                    f"{len(missing_ids)} DB records are missing — recovering..."
                )
                missing_df = labeled_df[
                    labeled_df["comment_id"].isin(missing_ids)
                ]
                recovery_records = [
                    {
                        "comment_id":   r["comment_id"],
                        "intent":       r["intent"],
                        "label_source": r.get("source"),
                        "latency_ms":   r.get("latency_ms"),
                        "cost_usd":     r.get("cost_usd"),
                    }
                    for _, r in missing_df.iterrows()
                ]
                inserted = label_repo.insert_labels_batch(
                    records=recovery_records,
                    model_name=self.model_name,
                    version=self.version,
                )
                self.logger.info(
                    f"[{PipelineEnum.LABELING}] Recovery complete — "
                    f"{inserted} DB records inserted."
                )

            except Exception as e:
                # DB recovery is best-effort — never block the pipeline
                self.logger.warning(
                    f"[{PipelineEnum.LABELING}] DB recovery failed (non-fatal): {e}"
                )
            return

        # ── LEVEL 3: fresh run ────────────────────────────────────────────────
        self.logger.info(f"[{PipelineEnum.LABELING}] Labeling with LLM...")
        df = self.storage.load_df(stage="processed_data", filename="clean_data.csv")
        comment_ids = df["comment_id"].dropna().astype(str).tolist() \
                    if "comment_id" in df.columns else []

        # Pre-fetch any existing labels from DB to skip re-labeling
        existing_labels = {}
        try:

            # Pass 1: same version (exact match)
            existing_labels = label_repo.get_labels_for_comments(
                comment_ids=comment_ids,
                model_name=self.model_name,
                version=self.version,
            )

            # Pass 2: fill gaps from older versions with the same LLM source
            missing_ids = [cid for cid in comment_ids if cid not in existing_labels]
            if missing_ids:
                cross_version_labels = label_repo.get_labels_for_comments_any_version(
                    comment_ids=missing_ids,
                    model_name=self.model_name,
                    label_source=self.llm_provider.generation_model_id,  # e.g. "gpt-4o-mini"
                    exclude_version=self.version,
                )
                self.logger.info(
                    f"[{PipelineEnum.LABELING}] Cross-version cache hit: "
                    f"{len(cross_version_labels)} labels reused from older versions."
                )
                existing_labels.update(cross_version_labels)

        except Exception as e:
            # DB lookup failure is non-fatal — just re-label everything
            self.logger.warning(
                f"[{PipelineEnum.LABELING}] DB pre-fetch failed (non-fatal): {e}. "
                "Proceeding with full LLM labeling."
            )

        labeled_df, new_db_records, cost_tracker = DataLabeling(
            clean_data=df,
            llm=self.llm_provider,
            existing_labels=existing_labels,
        ).generate_comment_intent()

        # ── Save CSV FIRST — DB write is secondary ────────────────────────────
        self.storage.save_df(df=labeled_df, stage="processed_data", filename="label_data.csv")
        self.storage.save_json(
            payload={
                "cost_tracker": cost_tracker,
                "rows":         len(labeled_df),
                "model_name":   self.model_name,
                "version":      self.version,
                "stage":        "labeling",
            },
            stage="processed_data",
            filename="label_data_metadata.json",
        )
        
        self.logger.info(
            f"[{PipelineEnum.LABELING}] CSV saved — "
            f"{len(labeled_df)} rows."
        )

        # ── Write new labels to DB (after CSV is safe) ────────────────────────
        if new_db_records:
            clean_records = [r for r in new_db_records if r.get("intent") is not None]

            try:
                inserted = label_repo.insert_labels_batch(
                    records=clean_records,
                    model_name=self.model_name,
                    version=self.version,
                )
                self.logger.info(
                    f"[{PipelineEnum.LABELING}] {inserted} new labels saved to DB."
                )
            except Exception as e:
                # Non-fatal — CSV is already saved, recovery runs on next trigger
                self.logger.warning(
                    f"[{PipelineEnum.LABELING}] DB write failed (non-fatal): {e}. "
                    "Will recover on next pipeline run."
                )

        self.logger.info(
            f"[{PipelineEnum.LABELING}] Done — "
            f"rows={len(labeled_df)}, "
            f"cached={cost_tracker['cached_count']}, "
            f"llm={cost_tracker['llm_count']}, "
            f"cost=${cost_tracker['total_cost']:.4f}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4 — DATA TRANSFORMATION
    # ─────────────────────────────────────────────────────────────────────────

    def data_transformation(self) -> None:
        if self._cached("processed_data", "data_transformation.csv"):
            return

        self.logger.info(f"[{PipelineEnum.TRANSFORMATION}] Transforming data...")
        df             = self.storage.load_df(stage="processed_data", filename="label_data.csv")
        transformed_df = DataTransformation(df).prepare_llm_finetuning_data()

        self.storage.save_df(
            df=transformed_df,
            stage="processed_data",
            filename="data_transformation.csv",
        )
        self.logger.info(
            f"[{PipelineEnum.TRANSFORMATION}] Done — {len(transformed_df)} rows saved."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5 — PREPARE TRAINING ASSETS
    # ─────────────────────────────────────────────────────────────────────────

    def prepare_training_assets(
        self,
        overrides: dict | None = None,
    ) -> None:

        if (
            self._cached("training/training_data",    "train.jsonl")
            and self._cached("training/training_data",    "val.jsonl")
            and self._cached("training/training_data",    "dataset_info.json")
            and self._cached("training/training_configs", "final_config.yaml")
        ):
            return

        self.logger.info(f"[{PipelineEnum.PRE_TRAINING}] Preparing training assets...")
        df     = self.storage.load_df(stage="processed_data", filename="data_transformation.csv")

        # overrides = only what the caller wants to change this run
        config = load_training_config()

        trainer          = ModelTraining(df=df, config=config, project=self.model_name, version=self.version)
        train_df, val_df = trainer.split_data()

        self.storage.save_df(df=train_df, stage="training/training_data", filename="train.jsonl")
        self.storage.save_df(df=val_df,   stage="training/training_data", filename="val.jsonl")

        if hasattr(trainer, "_split_report"):
            self.storage.save_json(
                payload=trainer._split_report,
                stage="training/training_data",
                filename="split_report.json",
            )
            self.logger.info(
                f"[{PipelineEnum.PRE_TRAINING}] Split — "
                f"train={trainer._split_report['train']} | "
                f"val={trainer._split_report['val']} | "
                f"stratified={trainer._split_report['stratified']}"
            )

        dataset_info = trainer.register_dataset_for_llmfactory()
        self.storage.save_json(
            payload=dataset_info,
            stage="training/training_data",
            filename="dataset_info.json",
        )

        # default YAML + optional per-run overrides
        final_config = trainer.build_llm_training_config(
            default_config=config,
            user_overrides=overrides,   # None = use YAML as-is
        )
        self.storage.save_yaml(
            payload=final_config,
            stage="training/training_configs",
            filename="final_config.yaml",
        )

        # audit: save exactly what overrides were applied
        self.storage.save_json(
            payload={
                "overrides_applied": overrides or {},
                "version":           self.version,
                "model_name":        self.model_name,
            },
            stage="training/training_configs",
            filename="overrides.json",
        )

        self.logger.info(
            f"[{PipelineEnum.PRE_TRAINING}] All training assets ready. "
            f"Overrides: {overrides or 'none (using YAML defaults)'}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6 — GPU TRAINING + EVALUATION
    # ─────────────────────────────────────────────────────────────────────────

    def run_model_training(self, db=None) -> None:
        self.logger.info(f"[{PipelineEnum.TRAINING}] ===== GPU TRAINING START =====")

        workspace  = f"/workspace/{self.model_name}/{self.version}"
        data_dir   = f"{workspace}/data"
        config_dir = f"{workspace}/config"
        output_dir = f"{workspace}/output"

        os.makedirs(data_dir,   exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # ── Check if model already trained ───────────────────────────────────
        # If adapter exists in S3, skip training and go straight to evaluation
        model_already_trained = self.storage.exists(
            "training/model_output", "adapter_config.json"
        )

        model_id  = None
        run_id    = None

        if model_already_trained:
            self.logger.info(
                f"[{PipelineEnum.TRAINING}] Model adapter found in S3 — "
                "skipping training, proceeding to evaluation."
            )
            # Download adapter for evaluation
            self.storage.download_to_disk(
                stage="training/model_output",
                filename="adapter_config.json",
                abs_path=f"{output_dir}/adapter_config.json",
            )
            # Try to recover model_id from DB
            if db is not None:
                try:
                    existing = ModelRepository(db).get_by_version(
                        model_name=self.model_name,
                        model_version=self.version,
                    )
                    if existing:
                        model_id = existing.id
                        run_id   = existing.mlflow_run_id
                        self.logger.info(
                            f"[{PipelineEnum.TRAINING}] Recovered model_id={model_id} from DB."
                        )
                except Exception as e:
                    self.logger.warning(f"[{PipelineEnum.TRAINING}] DB model lookup failed: {e}")

        else:
            # ── Download training files ───────────────────────────────────────
            self.logger.info(f"[{PipelineEnum.TRAINING}] Downloading training files from S3...")
            for filename in ("train.jsonl", "val.jsonl", "dataset_info.json"):
                self.storage.download_to_disk(
                    stage="training/training_data",
                    filename=filename,
                    abs_path=f"{data_dir}/{filename}",
                )

            yaml_config                = self.storage.load_yaml(
                stage="training/training_configs",
                filename="final_config.yaml",
            )
            yaml_config["dataset_dir"] = data_dir
            yaml_config["output_dir"]  = output_dir

            config_path = f"{config_dir}/final_config.yaml"
            self.storage.write_yaml_to_disk(yaml_config, config_path)

            # ── Train ─────────────────────────────────────────────────────────
            trainer = ModelTraining(
                df=None,
                config=yaml_config,
                project=self.model_name,
                version=self.version,
                storage=self.storage,
            )
            training_result = trainer.run_llamafactory_training(
                config_path=config_path,
                mlflow_tracking_uri=self.config.MLFLOW_TRACKING_URI,
                db=db,
            )

            run_id   = training_result["run_id"]
            model_id = training_result["model_id"]
            self.logger.info(
                f"[{PipelineEnum.TRAINING}] Training complete — "
                f"run_id={run_id} | model_id={model_id}"
            )

            # ── Upload adapter to S3 ──────────────────────────────────────────
            self.logger.info(f"[{PipelineEnum.TRAINING}] Uploading adapter to S3...")
            self.storage.upload_folder_to_s3(
                local_dir=output_dir,
                stage="training/model_output",
            )

        # ── Evaluation ────────────────────────────────────────────────────────
        self.logger.info(f"[{PipelineEnum.EVALUATION}] Starting evaluation...")

        # Check benchmark is available before doing anything
        benchmark_available = self.storage.exists("benchmark", "benchmark.csv")
        if not benchmark_available:
            self.logger.warning(
                f"[{PipelineEnum.EVALUATION}] Benchmark not available at "
                "benchmark/benchmark.csv — skipping evaluation."
            )
            return

        benchmark_df = self.storage.load_df(stage="benchmark", filename="benchmark.csv")

        # Validate benchmark columns
        required_cols = {"comment", "intent"}
        if not required_cols.issubset(benchmark_df.columns):
            self.logger.warning(
                f"[{PipelineEnum.EVALUATION}] Benchmark missing columns "
                f"{required_cols - set(benchmark_df.columns)} — skipping."
            )
            return

        self.logger.info(
            f"[{PipelineEnum.EVALUATION}] Benchmark loaded — {len(benchmark_df)} rows."
        )

        # Download full adapter folder for evaluation
        ADAPTER_FILES = [
            "adapter_config.json",
            "adapter_model.safetensors",
            "tokenizer_config.json",
            "tokenizer.json",
            "tokenizer.model",
            "special_tokens_map.json",
        ]

        for filename in ADAPTER_FILES:
            local_path = f"{output_dir}/{filename}"
            if not os.path.exists(local_path):
                try:
                    self.storage.download_to_disk(
                        stage="training/model_output",
                        filename=filename,
                        abs_path=local_path,
                    )
                except FileNotFoundError:
                    # Some files are optional depending on the model/tokenizer
                    self.logger.warning(
                        f"[{PipelineEnum.EVALUATION}] Optional adapter file not found: {filename} — skipping."
                    )

        evaluator = ModelEvaluation(
            adapter_path=output_dir,
            base_model=self.config.MODEL_NAME,
            benchmark_df=benchmark_df,
        )
        metrics = evaluator.evaluate()

        # ── Save evaluation to S3 ─────────────────────────────────────────────
        self.storage.save_json(
            payload=metrics,
            stage="training/evaluation",
            filename="evaluation_metrics.json",
        )
        self.logger.info(
            f"[{PipelineEnum.EVALUATION}] "
            f"accuracy={metrics['accuracy']} | f1={metrics['f1_macro']}"
        )

        # ── Save evaluation to DB ─────────────────────────────────────────────
        if db is not None and model_id is not None:
            try:
                eval_id = ModelEvaluationRepository(db).create_evaluation(
                    model_id=model_id,
                    accuracy=metrics["accuracy"],
                    f1_score=metrics["f1_macro"],
                    evaluation_dataset_hash=None,
                )
                self.logger.info(
                    f"[{PipelineEnum.EVALUATION}] Saved | "
                    f"eval_id={eval_id} | model_id={model_id}"
                )
            except Exception as e:
                self.logger.warning(f"[{PipelineEnum.EVALUATION}] DB write failed: {e}")
                if db:
                    db.rollback()
        elif model_id is None:
            self.logger.warning(
                f"[{PipelineEnum.EVALUATION}] model_id is None — "
                "evaluation metrics not saved to DB."
            )