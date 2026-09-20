import json
from datetime import datetime
from pathlib import Path

HISTORY_PATH = Path(__file__).parent / "models" / "training_history.json"


def _load_all() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    with open(HISTORY_PATH) as f:
        return json.load(f)


def _save_all(history: list[dict]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def append_entry(model_type: str, hyperparameters: dict, metrics: dict) -> dict:
    entry = {
        "trained_at": datetime.utcnow().isoformat(),
        "model_type": model_type,
        "hyperparameters": hyperparameters,
        "metrics": metrics,
    }
    history = _load_all()
    history.append(entry)
    _save_all(history)
    return entry


def get_history(model_type: str | None = None, limit: int | None = None) -> list[dict]:
    history = list(reversed(_load_all()))
    if model_type is not None:
        history = [entry for entry in history if entry["model_type"] == model_type]
    if limit is not None:
        history = history[:limit]
    return history
