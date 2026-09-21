import pandas as pd
from fastapi import APIRouter, Body, HTTPException

import state
from logging_config import logger
from openapi_examples import PREDICT_BODY_EXAMPLES, PREDICT_ERROR_RESPONSES
from preprocessing import select_and_order_features
from schemas import FeatureVectorChurn, PredictionResponseChurn

router = APIRouter()


@router.post("/predict", responses=PREDICT_ERROR_RESPONSES)
def predict(
    features: FeatureVectorChurn | list[FeatureVectorChurn] = Body(
        ...,
        openapi_examples=PREDICT_BODY_EXAMPLES,
    ),
) -> PredictionResponseChurn | list[PredictionResponseChurn]:
    model_record = state.get_model_record()
    if model_record is None:
        logger.error("Predict called but no trained model is available")
        raise HTTPException(
            status_code=400,
            detail="Model is not trained yet. Call POST /model/train first.",
        )

    is_batch = isinstance(features, list)
    items = features if is_batch else [features]

    logger.info("Predict requested for %d client(s)", len(items))

    raw_df = pd.DataFrame([item.model_dump() for item in items])
    X = select_and_order_features(raw_df)

    pipeline = model_record["pipeline"]
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)
    churn_col = list(pipeline.classes_).index(1)
    not_churn_col = list(pipeline.classes_).index(0)

    results = [
        PredictionResponseChurn(
            churn_prediction=int(pred),
            churn_probability=float(proba[churn_col]),
            not_churn_probability=float(proba[not_churn_col]),
        )
        for pred, proba in zip(predictions, probabilities)
    ]

    return results if is_batch else results[0]
