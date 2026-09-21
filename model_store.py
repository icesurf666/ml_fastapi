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
    model_type: str
    hyperparameters: dict


def save_churn_model(
    pipeline: Pipeline,
    metrics: dict,
    model_type: str,
    hyperparameters: dict,
    path: Path | None = None,
) -> ModelRecord:
    path = path or MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    record: ModelRecord = {
        "pipeline": pipeline,
        "metrics": metrics,
        "trained_at": datetime.utcnow().isoformat(),
        "model_type": model_type,
        "hyperparameters": hyperparameters,
    }
    joblib.dump(record, path)
    return record


def load_churn_model(path: Path | None = None) -> ModelRecord | None:
    path = path or MODEL_PATH
    if not path.exists():
        return None
    return joblib.load(path)
