import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    prepare_features,
    split_dataset,
)
from schemas import TrainingConfigChurn

MODEL_TYPES = {
    "logreg": LogisticRegression,
    "random_forest": RandomForestClassifier,
}


def build_classifier(model_type: str, hyperparameters: dict):
    if model_type not in MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Available: {list(MODEL_TYPES)}"
        )
    return MODEL_TYPES[model_type](**hyperparameters)


def build_pipeline(config: TrainingConfigChurn | None = None) -> Pipeline:
    config = config or TrainingConfigChurn()
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    classifier = build_classifier(config.model_type, config.hyperparameters)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )


def train_churn_model(
    df: pd.DataFrame,
    config: TrainingConfigChurn | None = None,
) -> tuple[Pipeline, dict]:
    config = config or TrainingConfigChurn()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    pipeline = build_pipeline(config)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    churn_col = list(pipeline.classes_).index(1)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba[:, churn_col]),
    }
    return pipeline, metrics
