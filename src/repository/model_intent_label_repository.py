from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from src.db.models.model_intent_label import ModelIntentLabel


class ModelIntentLabelRepository:

    def __init__(self, db: Session):
        self.db = db

    def get_labels_for_comments(
        self,
        comment_ids: list[str],
        model_name:  str,
        version:     str,
    ) -> dict[str, ModelIntentLabel]:
        """
        Bulk-fetch existing labels for the given comment_ids.

        Returns a dict keyed by comment_id for O(1) lookup in DataLabeling.
        Only returns labels that match model_name + version so re-labeling
        with a different model/version always produces fresh labels.
        """
        if not comment_ids:
            return {}

        rows = (
            self.db.query(ModelIntentLabel)
            .filter(
                ModelIntentLabel.comment_id.in_(comment_ids),
                ModelIntentLabel.model_name == model_name,
                ModelIntentLabel.version    == version,
            )
            .all()
        )

        # get_labels_for_comments
        return {str(row.comment_id): row for row in rows}


    def get_labels_for_comments_any_version(
        self,
        comment_ids:  list[str],
        model_name:   str,
        label_source: str,        # e.g. "gpt-4o-mini"
        exclude_version: str,     # don't re-use the current version (already handled)
    ) -> dict[str, ModelIntentLabel]:
        """
        Fallback lookup: find labels from older versions labeled by the same source.
        Used to avoid re-calling the LLM when the label is already trustworthy.
        """
        if not comment_ids:
            return {}

        rows = (
            self.db.query(ModelIntentLabel)
            .filter(
                ModelIntentLabel.comment_id.in_(comment_ids),
                ModelIntentLabel.model_name  == model_name,
                ModelIntentLabel.label_source == label_source,
                ModelIntentLabel.version     != exclude_version,
            )
            .order_by(ModelIntentLabel.version.desc())  # prefer most recent version
            .all()
        )


        # get_labels_for_comments_any_version
        seen = {}
        for row in rows:
            key = str(row.comment_id)
            if key not in seen:
                seen[key] = row
        
        return seen


    def insert_labels_batch(
        self,
        records:    list[dict],
        model_name: str,
        version:    str,
    ) -> int:
        """
        Bulk-insert new labels.
        ON CONFLICT DO NOTHING — idempotent, safe to retry after crashes.

        Each record dict must have:
            comment_id, intent, label_source, latency_ms, cost_usd

        Returns number of rows actually inserted.
        """
        if not records:
            return 0

        rows = [
            {
                "comment_id":   r["comment_id"],
                "model_name":   model_name,
                "version":      version,
                "intent":       r["intent"],
                "label_source": r.get("label_source"),
                "latency_ms":   r.get("latency_ms"),
                "cost_usd":     r.get("cost_usd"),
            }
            for r in records
        ]

        stmt = (
            insert(ModelIntentLabel)
            .values(rows)
            .on_conflict_do_nothing(
                constraint="uq_comment_model_version"
            )
        )

        result = self.db.execute(stmt)
        self.db.commit()
        return result.rowcount