import time
import json
import pandas as pd

from src.store.llm.providers.openai_provider import OpenAIProvider
from src.store.llm.llm_enum import OpenAIEnum
from src.store.llm.templates.comment_intent import SYSTEM_RULES, USER_RULES
from src.core.logger import pipeline_logger

logger = pipeline_logger


VALID_LABELS = {"Question", "Complaint", "Statement", "Praise", "Suggestion"}


class DataLabeling:

    def __init__(
        self,
        clean_data:     pd.DataFrame,
        llm:            OpenAIProvider,
        existing_labels: dict[int, object] | None = None,
    ):
        """
        Args:
            clean_data      : DataFrame from DataClean — must have 'comment', 'comment_id'
            llm             : LLM provider (OpenAI)
            existing_labels : {comment_id: ModelIntentLabel} pre-fetched from DB.
                              Pass None or {} to force LLM labeling for all rows.
        """
        self.clean_data      = clean_data
        self.llm             = llm
        self.existing_labels = existing_labels or {}

        self.cost_tracker = {
            "total_cost":       0.0,
            "llm_cost":         0.0,
            "llm_count":        0,
            "cached_count":     0,   # rows that came from DB cache
            "total_tokens_in":  0,
            "total_tokens_out": 0,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_valid_intent(self, raw_text: str, comment: str) -> str:
        """Parse LLM JSON response and return a valid intent label."""
        try:
            cleaned = raw_text.replace("```json", "").replace("```", "").strip()
            parsed  = json.loads(cleaned)
            intent  = parsed.get("predicted_intent")

            if isinstance(intent, str):
                intent = intent.strip().capitalize()

            if intent not in VALID_LABELS:
                logger.warning(f"[DataLabeling] Invalid label '{intent}' → fallback to Statement")
                return "Statement"

            return intent

        except Exception:
            logger.warning(f"[DataLabeling] JSON parse failed for: {comment[:50]!r} → fallback")
            return "Statement"

    # ── Main ──────────────────────────────────────────────────────────────────

    def generate_comment_intent(self) -> tuple[pd.DataFrame, list[dict], dict]:
        """
        Label every comment in clean_data.

        Returns:
            labeled_df   : DataFrame with columns [comment, intent, source,
                           confidence, latency_ms, cost_usd, stage]
            new_db_records : list of dicts for rows that were LLM-labeled
                             (not cached) — caller saves these to DB
            cost_tracker : summary of cost and counts
        """
        predictions    = []
        new_db_records = []

        for _, row in self.clean_data.iterrows():
            comment    = row["comment"]
            comment_id = row.get("comment_id")

            # ── PATH A: existing label in DB ──────────────────────────────
            cached = self.existing_labels.get(comment_id)
            if cached is not None:
                predictions.append({
                    "comment_id": comment_id,
                    "comment":    comment,
                    "intent":     cached.intent,
                    "source":     cached.label_source,
                    "confidence": 1.0,
                    "latency_ms": cached.latency_ms,
                    "cost_usd":   0.0,   # already paid for
                    "stage":      "training",
                })
                self.cost_tracker["cached_count"] += 1
                continue

            # ── PATH B: call LLM ──────────────────────────────────────────
            start = time.time()

            messages = [
                self.llm.construct_prompt(SYSTEM_RULES.safe_substitute(), role=OpenAIEnum.SYSTEM.value),
                self.llm.construct_prompt(USER_RULES.substitute(comment=comment), role=OpenAIEnum.USER.value),
            ]
            response   = self.llm.generate_text(chat_history=messages)
            latency_ms = (time.time() - start) * 1000

            intent            = self._extract_valid_intent(response["text"], comment)
            prompt_tokens     = response.get("prompt_tokens", 0)
            completion_tokens = response.get("completion_tokens", 0)
            cost_usd          = self.llm.estimate_cost(prompt_tokens, completion_tokens)

            # update cost tracker
            self.cost_tracker["llm_cost"]         += cost_usd
            self.cost_tracker["total_cost"]        += cost_usd
            self.cost_tracker["llm_count"]         += 1
            self.cost_tracker["total_tokens_in"]   += prompt_tokens
            self.cost_tracker["total_tokens_out"]  += completion_tokens

            predictions.append({
                "comment":    comment,
                "intent":     intent,
                "source":     self.llm.generation_model_id,
                "confidence": 1.0,
                "latency_ms": round(latency_ms, 2),
                "cost_usd":   round(cost_usd, 6),
                "stage":      "training",
            })

            # only new LLM-labeled rows go to DB — cached rows are already there
            new_db_records.append({
                "comment_id":   comment_id,
                "intent":       intent,
                "label_source": self.llm.generation_model_id,
                "latency_ms":   round(latency_ms, 2),
                "cost_usd":     round(cost_usd, 6),
            })

        logger.info(
            f"[DataLabeling] Done — "
            f"cached={self.cost_tracker['cached_count']}, "
            f"llm={self.cost_tracker['llm_count']}, "
            f"cost=${self.cost_tracker['total_cost']:.4f}"
        )

        return pd.DataFrame(predictions), new_db_records, self.cost_tracker