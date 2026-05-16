import json
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


from src.core.logger import pipeline_logger

logger = pipeline_logger

VALID_LABELS = {"Complaint", "Question", "Suggestion", "Statement", "Praise"}


class ModelEvaluation:

    def __init__(
        self,
        adapter_path:   str,
        base_model:     str           = "Qwen/Qwen2.5-1.5B-Instruct",
        benchmark_df:   pd.DataFrame  = None,
        benchmark_path: str           = None,
    ):
        """
        Args:
            adapter_path   : Local path to the LoRA adapter (output_dir from training)
            base_model     : HuggingFace model ID used during training
            benchmark_df   : DataFrame with columns [comment, intent]
            benchmark_path : Path to benchmark CSV (fallback if benchmark_df is None)
        """
        self.adapter_path   = adapter_path
        self.base_model     = base_model
        self.benchmark_df   = benchmark_df
        self.benchmark_path = benchmark_path
        self._pipeline      = None   # lazy-loaded — avoids loading GPU model unless needed

    # ─────────────────────────────────────────────────────────────────────────
    # DATA LOADING
    # ─────────────────────────────────────────────────────────────────────────

    def load_benchmark(self) -> pd.DataFrame:
        if self.benchmark_df is not None:
            logger.info(f"[ModelEvaluation] Using in-memory benchmark ({len(self.benchmark_df)} rows)")
            return self.benchmark_df

        if self.benchmark_path:
            logger.info(f"[ModelEvaluation] Loading benchmark from: {self.benchmark_path}")
            return pd.read_csv(self.benchmark_path)

        raise ValueError(
            "ModelEvaluation requires either benchmark_df or benchmark_path."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL LOADING (lazy)
    # ─────────────────────────────────────────────────────────────────────────

    def _load_pipeline(self) -> None:
        """Lazy-load the fine-tuned model — only runs once."""
        if self._pipeline is not None:
            return

        logger.info(f"[ModelEvaluation] Loading base model: {self.base_model}")
        logger.info(f"[ModelEvaluation] Loading adapter:    {self.adapter_path}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, self.adapter_path)
        model.eval()

        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=32,
            do_sample=False,
        )

        logger.info("[ModelEvaluation] Model loaded and ready.")

    # ─────────────────────────────────────────────────────────────────────────
    # INFERENCE
    # ─────────────────────────────────────────────────────────────────────────

    def predict(self, comment: str) -> str:
        """
        Run inference on a single comment.
        Returns one of: Complaint | Question | Suggestion | Statement | Praise
        Falls back to 'Statement' on any parse failure.
        """
        self._load_pipeline()

        prompt = (
            f"Classify the intent of the following comment.\n"
            f"Return JSON only: {{\"predicted_intent\": \"<label>\"}}\n\n"
            f"Comment: {comment}"
        )

        output    = self._pipeline(prompt)[0]["generated_text"]
        generated = output[len(prompt):].strip()

        try:
            parsed = json.loads(generated)
            intent = parsed.get("predicted_intent", "").strip().capitalize()
            if intent not in VALID_LABELS:
                logger.warning(
                    f"[ModelEvaluation] Invalid prediction '{intent}' "
                    f"for: {comment[:50]!r} → fallback to Statement"
                )
                return "Statement"
            return intent
        except Exception:
            logger.warning(
                f"[ModelEvaluation] JSON parse failed for: {comment[:50]!r} "
                "→ fallback to Statement"
            )
            return "Statement"

    # ─────────────────────────────────────────────────────────────────────────
    # EVALUATION LOOP
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(self) -> dict:
        """
        Run full evaluation against the benchmark.

        Returns
        ───────
        {
            "accuracy":    float,
            "f1_macro":    float,
            "f1_weighted": float,
            "total":       int,
            "correct":     int,
            "report":      str,   # full sklearn classification report
        }
        """
        df = self.load_benchmark()

        # ── Validate columns ──────────────────────────────────────────────
        required = {"comment", "intent"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(
                f"[ModelEvaluation] Benchmark is missing columns: {missing}. "
                f"Got: {list(df.columns)}"
            )

        # ── Drop nulls ────────────────────────────────────────────────────
        before = len(df)
        df     = df.dropna(subset=["comment", "intent"]).reset_index(drop=True)
        if len(df) < before:
            logger.warning(
                f"[ModelEvaluation] Dropped {before - len(df)} null rows from benchmark."
            )

        logger.info(f"[ModelEvaluation] Running evaluation on {len(df)} examples...")

        y_true: list[str] = []
        y_pred: list[str] = []

        for i, (_, row) in enumerate(df.iterrows()):
            pred = self.predict(str(row["comment"]))
            y_true.append(str(row["intent"]).strip().capitalize())
            y_pred.append(pred)

            if (i + 1) % 10 == 0:
                logger.info(f"[ModelEvaluation] Progress: {i + 1}/{len(df)}")

        # ── Metrics ───────────────────────────────────────────────────────
        accuracy    = accuracy_score(y_true, y_pred)
        f1_macro    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        correct     = sum(t == p for t, p in zip(y_true, y_pred))
        report      = classification_report(y_true, y_pred, zero_division=0)

        metrics = {
            "accuracy":    round(float(accuracy),    4),
            "f1_macro":    round(float(f1_macro),    4),
            "f1_weighted": round(float(f1_weighted), 4),
            "total":       len(y_true),
            "correct":     correct,
            "report":      report,
        }

        logger.info(
            f"[ModelEvaluation] "
            f"accuracy={metrics['accuracy']:.4f} | "
            f"f1_macro={metrics['f1_macro']:.4f} | "
            f"correct={correct}/{len(y_true)}"
        )
        logger.info(f"[ModelEvaluation] Classification Report:\n{report}")

        # ── Log to MLflow if a run is active ──────────────────────────────
        try:
            mlflow.log_metric("eval/accuracy",    metrics["accuracy"])
            mlflow.log_metric("eval/f1_macro",    metrics["f1_macro"])
            mlflow.log_metric("eval/f1_weighted", metrics["f1_weighted"])
            mlflow.log_metric("eval/correct",     metrics["correct"])
            mlflow.log_metric("eval/total",       metrics["total"])
            mlflow.log_text(report, "eval_classification_report.txt")
        except Exception:
            pass  # no active MLflow run — fine, metrics already saved to S3

        return metrics