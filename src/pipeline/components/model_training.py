import os
import subprocess
import time
import math
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Union

from src.core.logger import pipeline_logger

logger = pipeline_logger


class ModelTraining:
    """
    Handles all model training preparation and execution.

    Responsibilities:
        - Split data into train/val sets
        - Register dataset for LLaMA-Factory
        - Build training config YAML
        - Run LLaMA-Factory fine-tuning via CLI
        - Log everything to MLflow
    """

    LABEL_MAP = {
        "complaint":  "Complaint",
        "question":   "Question",
        "suggestion": "Suggestion",
        "statement":  "Statement",
        "praise":     "Praise",
    }

    def __init__(self, df: pd.DataFrame, config: dict, project: str, version: str, storage=None):
        """
        Args:
            df      : Transformed DataFrame with columns [instruction, input, output]
            config  : Training config dict loaded from YAML
            project : Model/project name  (e.g. "ladybug")
            version : Pipeline version    (e.g. "v1")
        """
        self.df      = df
        self.config  = config
        self.project = project
        self.version = version
        self.storage = storage 

    # ------------------------------------------------------------------ #
    #  DATA SPLIT                                                          #
    # ------------------------------------------------------------------ #

    def split_data(
        self,
        val_size:                  Union[float, int] = 0.1,
        random_state:              int  = 42,
        ensure_all_classes_in_val: bool = True,
        min_val_samples_per_class: int  = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        if "output" not in self.df.columns:
            raise ValueError("DataFrame must contain 'output' column.")
        if "intent" not in self.df.columns:
            raise ValueError("DataFrame must contain 'intent' column for stratification.")

        df = self.df.copy()

        label_counts = df["intent"].value_counts()
        logger.info(f"[ModelTraining] Full dataset — {len(df)} rows")
        logger.info(f"[ModelTraining] Class distribution:\n{label_counts.to_string()}")

        # ── Resolve val_size to absolute count ────────────────────────────────
        n_total = len(df)
        if isinstance(val_size, float):
            n_val = max(1, math.floor(n_total * val_size))
        else:
            n_val = int(val_size)

        if n_val >= n_total:
            raise ValueError(
                f"val_size={val_size} would consume the entire dataset ({n_total} rows)."
            )

        # ── Decide whether stratification is safe ────────────────────────────
        min_class_count = label_counts.min()
        n_classes       = label_counts.nunique()
        stratify_col    = None

        if min_class_count < 2:
            logger.warning(
                f"[ModelTraining] Rare class with only {min_class_count} sample(s) "
                "→ disabling stratification."
            )
        elif n_val < n_classes:
            logger.warning(
                f"[ModelTraining] val_size ({n_val}) < n_classes ({n_classes}) "
                "→ disabling stratification."
            )
        else:
            stratify_col = df["intent"]
            logger.info("[ModelTraining] Stratified split enabled.")

        # ── Split ─────────────────────────────────────────────────────────────
        train_df, val_df = train_test_split(
            df,
            test_size=n_val,
            random_state=random_state,
            stratify=stratify_col,
        )

        # ── Guarantee every class appears in val ─────────────────────────────
        if ensure_all_classes_in_val:
            missing_classes = set(df["intent"]) - set(val_df["intent"])
            if missing_classes:
                logger.warning(
                    f"[ModelTraining] {len(missing_classes)} class(es) missing from val "
                    f"→ {missing_classes} — moving samples from train."
                )
                for cls in missing_classes:
                    cls_in_train = train_df[train_df["intent"] == cls]
                    n_to_move    = min(min_val_samples_per_class, len(cls_in_train))

                    if n_to_move == 0:
                        logger.error(
                            f"[ModelTraining] Class '{cls}' has 0 samples in train — "
                            "cannot fix val. Check your dataset."
                        )
                        continue

                    sample   = cls_in_train.sample(n_to_move, random_state=random_state)
                    val_df   = pd.concat([val_df, sample], ignore_index=True)
                    train_df = train_df.drop(sample.index)

        # ── Split report ──────────────────────────────────────────────────────
        self._split_report = {
            "total":        n_total,
            "train":        len(train_df),
            "val":          len(val_df),
            "val_pct":      round(len(val_df) / n_total * 100, 2),
            "random_state": random_state,
            "stratified":   stratify_col is not None,
            "train_dist":   train_df["intent"].value_counts().to_dict(),
            "val_dist":     val_df["intent"].value_counts().to_dict(),
        }

        logger.info(
            f"[ModelTraining] Split complete — "
            f"train={len(train_df)} | val={len(val_df)} "
            f"({self._split_report['val_pct']}%)"
        )
        logger.info(f"[ModelTraining] Train dist: {self._split_report['train_dist']}")
        logger.info(f"[ModelTraining] Val dist:   {self._split_report['val_dist']}")

        # ── Drop intent column before returning ───────────────────────────────
        train_df = train_df.drop(columns=["intent"]).reset_index(drop=True)
        val_df   = val_df.drop(columns=["intent"]).reset_index(drop=True)

        return train_df, val_df



    # ------------------------------------------------------------------ #
    #  DATASET REGISTRATION                                                #
    # ------------------------------------------------------------------ #

    def register_dataset_for_llmfactory(self) -> dict:
        """
        Build the dataset_info.json entry required by LLaMA-Factory.
        Keys match the dataset names used in the training YAML config.
        """
        dataset_info = {
            f"{self.project}_{self.version}_train": {
                "file_name": "train.jsonl",
                "columns": {
                    "prompt":   "instruction",
                    "query":    "input",
                    "response": "output",
                },
            },
            f"{self.project}_{self.version}_val": {
                "file_name": "val.jsonl",
                "columns": {
                    "prompt":   "instruction",
                    "query":    "input",
                    "response": "output",
                },
            },
        }
        logger.info("[ModelTraining] dataset_info.json built")
        return dataset_info



    # ------------------------------------------------------------------ #
    #  YAML CONFIG BUILDER                                                 #
    # ------------------------------------------------------------------ #

    def build_llm_training_config(
        self,
        default_config: dict,
        user_overrides: dict | None = None,
    ) -> dict:
        config = dict(default_config)

        # ── Pipeline-injected values (always set, never overridable) ─────────
        config["dataset"]      = f"{self.project}_{self.version}_train"
        config["eval_dataset"] = f"{self.project}_{self.version}_val"
        config["report_to"]    = "none"
        config["output_dir"]   = f"/workspace/outputs/{self.project}/{self.version}"

        # ── User overrides win last ───────────────────────────────────────────
        if user_overrides:
            config.update(user_overrides)

        # ── Log the keys that actually affect training quality ────────────────
        logger.info(
            f"[ModelTraining] Final training config | "
            f"model={config.get('model_name_or_path')} | "
            f"epochs={config.get('num_train_epochs')} | "
            f"lr={config.get('learning_rate')} | "
            f"lora_rank={config.get('lora_rank')} | "
            f"batch={config.get('per_device_train_batch_size')} | "
            f"grad_accum={config.get('gradient_accumulation_steps')} | "
            f"overrides_applied={list(user_overrides.keys()) if user_overrides else 'none'}"
        )

        return config



    # ------------------------------------------------------------------ #
    #  TRAINING RUNNER                                                     #
    # ------------------------------------------------------------------ #

    def run_llamafactory_training(
        self,
        config_path:         str,
        mlflow_tracking_uri: str,
        db=None,
    ) -> dict:
        """
        Run LLaMA-Factory SFT training and log everything to MLflow.
        Reads split_report.json and label_data_metadata.json from S3
        to include full cost and data lineage in MLflow.
        """
        from src.repository.model_repository import ModelRepository

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(f"{self.project}-{self.version}")

        # ── GPU config from YAML ──────────────────────────────────────────────
        gpu_type          = self.config.get("gpu_type",             "unknown")
        gpu_cost_per_hour = float(self.config.get("gpu_cost_per_hour", 1.50))

        start_time = time.time()

        with mlflow.start_run() as run:
            run_id = run.info.run_id

            # ── Core params ───────────────────────────────────────────────────
            mlflow.log_param("project",       self.project)
            mlflow.log_param("version",       self.version)
            mlflow.log_param("model",         self.config.get("model_name_or_path", "unknown"))
            mlflow.log_param("lora_rank",     self.config.get("lora_rank",          "unknown"))
            mlflow.log_param("epochs",        self.config.get("num_train_epochs",   "unknown"))
            mlflow.log_param("learning_rate", self.config.get("learning_rate",      "unknown"))
            mlflow.log_param("batch_size",    self.config.get("per_device_train_batch_size", "unknown"))
            mlflow.log_param("gpu_type",      gpu_type)

            # ── Full config artifact ──────────────────────────────────────────
            mlflow.log_dict(self.config, "training_config.json")

            # ── Split report from S3 ──────────────────────────────────────────
            try:
                split_report = self.storage.load_json(
                    stage="training/training_data",
                    filename="split_report.json",
                )
                mlflow.log_dict(split_report, "split_report.json")
                mlflow.log_param("train_rows",  split_report.get("train"))
                mlflow.log_param("val_rows",    split_report.get("val"))
                mlflow.log_param("stratified",  split_report.get("stratified"))
            except Exception as e:
                logger.warning(f"[MLflow] Could not load split_report.json: {e}")
                split_report = {}

            # ── Labeling cost from S3 ─────────────────────────────────────────
            labeling_cost_usd = 0.0
            try:
                label_metadata = self.storage.load_json(
                    stage="processed_data",
                    filename="label_data_metadata.json",
                )
                cost_tracker      = label_metadata.get("cost_tracker", {})
                labeling_cost_usd = float(cost_tracker.get("total_cost", 0.0))
                mlflow.log_dict(label_metadata, "label_metadata.json")
                mlflow.log_metric("labeling_cost_usd",  labeling_cost_usd)
                mlflow.log_metric("labeling_llm_count", cost_tracker.get("llm_count", 0))
            except Exception as e:
                logger.warning(f"[MLflow] Could not load label_data_metadata.json: {e}")

            # ── Run LLaMA-Factory ─────────────────────────────────────────────
            env = os.environ.copy()
            env["WANDB_DISABLED"] = "true"

            process = subprocess.Popen(
                ["llamafactory-cli", "train", config_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )

            for line in process.stdout:
                logger.info(line.rstrip())

            process.wait()

            if process.returncode != 0:
                mlflow.set_tag("status", "failed")
                raise RuntimeError(f"LLaMA-Factory exited with code {process.returncode}")

            # ── GPU time + cost ───────────────────────────────────────────────
            duration_sec  = time.time() - start_time
            gpu_hours     = duration_sec / 3600
            gpu_cost_usd  = gpu_hours * gpu_cost_per_hour
            total_cost    = round(gpu_cost_usd + labeling_cost_usd, 6)

            mlflow.log_metric("training_time_sec", round(duration_sec, 2))
            mlflow.log_metric("gpu_hours",         round(gpu_hours, 4))
            mlflow.log_metric("gpu_cost_usd",      round(gpu_cost_usd, 4))
            mlflow.log_metric("labeling_cost_usd", labeling_cost_usd)
            mlflow.log_metric("total_cost_usd",    total_cost)
            mlflow.set_tag("status", "success")

            logger.info(
                f"[MLflow] run_id={run_id} | "
                f"duration={duration_sec:.0f}s | "
                f"gpu_cost=${gpu_cost_usd:.3f} | "
                f"labeling_cost=${labeling_cost_usd:.4f} | "
                f"total=${total_cost:.4f}"
            )

            # ── Save model to DB ──────────────────────────────────────────────
            model_id = None
            if db is not None:
                try:
                    model_id = ModelRepository(db).create_model_run(
                        model_name=self.project,
                        model_version=self.version,
                        mlflow_run_id=run_id,
                        gpu_type=gpu_type,
                        gpu_hours=round(gpu_hours, 4),
                        total_training_cost_usd=total_cost,
                    )
                    logger.info(f"[DB] Model saved | model_id={model_id} | run_id={run_id}")
                except Exception as e:
                    logger.error(f"[DB] Failed to save model: {e}")
                    if db:
                        db.rollback()

            return {
                "run_id":         run_id,
                "model_id":       model_id,
                "gpu_hours":      round(gpu_hours, 4),
                "gpu_cost_usd":   round(gpu_cost_usd, 4),
                "labeling_cost":  labeling_cost_usd,
                "total_cost_usd": total_cost,
            }