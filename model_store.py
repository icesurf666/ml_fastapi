from datetime import datetime
from pathlib import Path
from typing import TypedDict

import joblib
from sklearn.pipeline import Pipeline

MODEL_PATH = Path(__file__).parent / "models" / "churn_model.joblib"


class ModelRecord(TypedDict):
    pipeline: Pipeline
    metrics: dict
    trained_at: str


def save_churn_model(pipeline: Pipeline, metrics: dict, path: Path = MODEL_PATH) -> ModelRecord:
    path.parent.mkdir(parents=True, exist_ok=True)
    record: ModelRecord = {
        "pipeline": pipeline,
        "metrics": metrics,
        "trained_at": datetime.utcnow().isoformat(),
    }
    joblib.dump(record, path)
    return record


def load_churn_model(path: Path = MODEL_PATH) -> ModelRecord | None:
    if not path.exists():
        return None
    return joblib.load(path)
