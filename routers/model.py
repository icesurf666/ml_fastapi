from fastapi import APIRouter, HTTPException

import state
from model import train_churn_model
from model_store import save_churn_model
from openapi_examples import TRAIN_ERROR_RESPONSES
from preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from schemas import FeatureVectorChurn, TrainingConfigChurn
from training_history import append_entry, get_history

router = APIRouter(prefix="/model")


@router.post("/train", responses=TRAIN_ERROR_RESPONSES)
def model_train(config: TrainingConfigChurn | None = None):
    config = config or TrainingConfigChurn()

    if state.dataset_df is None or state.dataset_df.empty:
        raise HTTPException(status_code=400, detail="Dataset is not loaded or empty")

    try:
        pipeline, metrics = train_churn_model(state.dataset_df, config)
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error))

    record = save_churn_model(
        pipeline,
        metrics,
        model_type=config.model_type,
        hyperparameters=config.hyperparameters,
    )
    state.set_model_record(record)
    append_entry(
        model_type=config.model_type,
        hyperparameters=config.hyperparameters,
        metrics=metrics,
    )
    return metrics


@router.get("/schema")
def model_schema():
    field_types = {
        name: field.annotation.__name__
        for name, field in FeatureVectorChurn.model_fields.items()
    }
    return {
        "features": list(field_types.keys()),
        "feature_types": field_types,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }


@router.get("/metrics")
def model_metrics(limit: int = 5, model_type: str | None = None):
    history = get_history(model_type=model_type, limit=limit)
    return {
        "latest": history[0] if history else None,
        "history": history,
    }


@router.get("/status")
def model_status():
    model_record = state.get_model_record()
    if model_record is None:
        return {
            "trained": False,
            "trained_at": None,
            "metrics": None,
            "model_type": None,
            "hyperparameters": None,
        }

    return {
        "trained": True,
        "trained_at": model_record["trained_at"],
        "metrics": model_record["metrics"],
        "model_type": model_record["model_type"],
        "hyperparameters": model_record["hyperparameters"],
    }
