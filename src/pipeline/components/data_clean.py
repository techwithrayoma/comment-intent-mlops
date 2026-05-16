import pandas as pd


class DataClean:

    MIN_CLEAN_LENGTH = 10

    def __init__(self, ingested_data: pd.DataFrame):
        self.df = ingested_data.copy()

    def clean_training_data(self) -> pd.DataFrame:
        initial_count = len(self.df)

        self.df = (
            self.df
            .pipe(self._rename_columns)
            .pipe(self._drop_nulls)
            .pipe(self._normalize_text)
            .pipe(self._drop_too_short)
            .pipe(self._drop_text_duplicates)
            .pipe(self._reset_index)
        )
        return self.df

    # ── steps ────────────────────────────────────────────────────────────────

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column name so downstream steps don't need to branch."""
        return df.rename(columns={"comment_text": "comment"}) \
                 if "comment_text" in df.columns else df

    def _drop_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna(subset=["comment"])

    def _normalize_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Strip leading/trailing whitespace and normalize to plain string.
        Intentionally minimal — no lowercasing, no punctuation removal.
        The LLM labeler and fine-tuning both benefit from natural casing.
        """
        df = df.copy()
        df["comment"] = df["comment"].astype(str).str.strip()
        return df

    def _drop_too_short(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop comments that are too short to carry training signal.
        These are typically emoji-only or single-word reactions ("lol", "nice").
        """
        mask = df["comment"].str.len() >= self.MIN_CLEAN_LENGTH
        dropped = (~mask).sum()
        if dropped:
            print(f"[DataClean] Dropped {dropped} comments below {self.MIN_CLEAN_LENGTH} chars after cleaning.")
        return df[mask]

    def _drop_text_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove comments with identical text (different comment_ids, same content).

        We keep the row with the earliest published_at so the most original
        version survives (not the copy).
        """
        before = len(df)

        # Sort so the earliest occurrence is kept
        if "published_at" in df.columns:
            df = df.sort_values("published_at", ascending=True)

        df = df.drop_duplicates(subset=["comment"], keep="first")
        dropped = before - len(df)

        if dropped:
            print(f"[DataClean] Dropped {dropped} text-duplicate comment(s).")

        return df

    def _reset_index(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.reset_index(drop=True)
