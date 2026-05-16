from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import text


class CommentRepository:
    """
    DB access for the ingestion step.

    Two responsibilities only:
        1. Fetch unseen comments that qualify for training.
        2. Mark a set of comment_ids as used (so future runs skip them).
    """

    def __init__(self, db_session: Session):
        self.db = db_session

    # ─────────────────────────────────────────────────────────────────────────
    # FETCH
    # ─────────────────────────────────────────────────────────────────────────

    def get_training_comments(
        self,
        model_name:      str,
        start_date:      datetime,
        end_date:        datetime,
        min_char_length: int  = 20,
        max_char_length: int  = 2000,
        max_rows:        int  = 50_000,
        exclude_seen:    bool = True,
    ) -> list[dict]:
        """
        Pull comments that:
            ① were published in the date window
            ② have a reasonable text length (not spam, not essays)
            ③ have NOT already been used to train this model  ← the key filter

        Why ③ matters
        ─────────────
        Without it, every training run would recycle the same old comments.
        The model would overfit to stale data and never learn from new comments.
        With it, v2 automatically gets only what v1 hasn't seen — no manual
        date tracking needed.
        """

        seen_clause = ""
        if exclude_seen:
            seen_clause = """
                AND c.id NOT IN (
                    SELECT comment_id
                    FROM   training_snapshots
                    WHERE  model_name = :model_name
                )
            """

        rows = self.db.execute(
            text(f"""
                SELECT
                    c.id           AS comment_id,
                    c.comment_text,
                    c.published_at
                FROM  comments c
                WHERE c.published_at   BETWEEN :start_date AND :end_date
                  AND LENGTH(c.comment_text) BETWEEN :min_char AND :max_char
                  {seen_clause}
                ORDER BY c.published_at DESC
                LIMIT :max_rows
            """),
            {
                "model_name": model_name,
                "start_date": start_date,
                "end_date":   end_date,
                "min_char":   min_char_length,
                "max_char":   max_char_length,
                "max_rows":   max_rows,
            },
        ).mappings().all()

        return [dict(r) for r in rows]

    # ─────────────────────────────────────────────────────────────────────────
    # MARK AS SEEN
    # ─────────────────────────────────────────────────────────────────────────

    def mark_comments_as_seen(
        self,
        model_name:  str,
        version:     str,
        comment_ids: list[int],
    ) -> int:
        """
        Insert one row per comment_id into training_snapshots.

        ON CONFLICT DO NOTHING means re-running after a crash is always safe.
        Returns the number of newly inserted rows.
        """
        if not comment_ids:
            return 0

        result = self.db.execute(
            text("""
                INSERT INTO training_snapshots (model_name, version, comment_id)
                SELECT :model_name, :version, UNNEST(CAST(:ids AS int[]))
                ON CONFLICT (model_name, comment_id) DO NOTHING
            """),
            {"model_name": model_name, "version": version, "ids": comment_ids},
        )
        self.db.commit()
        return result.rowcount