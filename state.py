from dataset import load_dataset
from model_store import ModelRecord, load_churn_model

dataset_df = load_dataset()
_model_record: ModelRecord | None = load_churn_model()


def get_model_record() -> ModelRecord | None:
    return _model_record


def set_model_record(record: ModelRecord) -> None:
    global _model_record
    _model_record = record
