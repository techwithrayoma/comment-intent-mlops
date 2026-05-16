import json
import pandas as pd

from src.pipeline.templates.finetune_data import INSTRUCTION_FINETUNE, SYSTEM_FINETUNE


class DataTransformation:
    """
    Transforms labeled comments into LLaMA-Factory SFT format.

    Input  columns: 'comment', 'intent'
    Output columns: 'system', 'instruction', 'input', 'output', 'history'
    """

    REQUIRED_COLUMNS = {"comment", "intent"}

    def __init__(self, labeled_data: pd.DataFrame):
        self.labeled_data = labeled_data

    def prepare_llm_finetuning_data(self) -> pd.DataFrame:
        df = self.labeled_data.copy()

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"[DataTransformation] Missing columns: {missing}")

        # Drop nulls / empty values
        df = df.dropna(subset=["comment", "intent"])
        df = df[df["comment"].astype(str).str.strip().str.len() > 0]
        df = df[df["intent"].astype(str).str.strip().str.len() > 0]
        df = df.reset_index(drop=True)

        system_message = SYSTEM_FINETUNE.substitute()

        records = [
            {
                "system":      system_message,
                "instruction": INSTRUCTION_FINETUNE.substitute(comment=str(row["comment"]).strip()),
                "input":       "",
                "output":      json.dumps(
                                    {"predicted_intent": str(row["intent"]).strip()},
                                    ensure_ascii=False,
                               ),
                "history":     [],
                "intent": row["intent"] 
            }
            for _, row in df.iterrows()
        ]

        result = pd.DataFrame(records)

        if result.empty:
            raise ValueError("[DataTransformation] Output DataFrame is empty after transformation.")

        return result