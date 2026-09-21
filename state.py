from dataset import load_dataset
from logging_config import logger
from model_store import ModelRecord, load_churn_model

dataset_df = load_dataset()
logger.info("Loaded churn dataset: %d rows, %d columns", *dataset_df.shape)

_model_record: ModelRecord | None = load_churn_model()
if _model_record is not None:
    logger.info(
        "Loaded existing model from disk: type=%s trained_at=%s",
        _model_record["model_type"],
        _model_record["trained_at"],
    )
else:
    logger.info("No existing trained model found on disk")


def get_model_record() -> ModelRecord | None:
    return _model_record


def set_model_record(record: ModelRecord) -> None:
    global _model_record
    _model_record = record
