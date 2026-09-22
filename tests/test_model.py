import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.ml.model import build_classifier, build_pipeline, train_churn_model
from app.schemas import TrainingConfigChurn


def test_build_classifier_logreg_returns_logistic_regression():
    classifier = build_classifier("logreg", {})

    assert isinstance(classifier, LogisticRegression)


def test_build_classifier_random_forest_applies_hyperparameters():
    classifier = build_classifier("random_forest", {"n_estimators": 10})

    assert isinstance(classifier, RandomForestClassifier)
    assert classifier.n_estimators == 10


def test_build_classifier_unknown_model_type_raises_value_error():
    with pytest.raises(ValueError):
        build_classifier("xgboost", {})


def test_build_pipeline_has_preprocessor_and_model_steps():
    pipeline = build_pipeline(TrainingConfigChurn())

    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["preprocessor", "model"]


def test_train_churn_model_returns_fitted_pipeline_and_metrics(sample_df):
    pipeline, metrics = train_churn_model(sample_df, TrainingConfigChurn())

    assert {"accuracy", "f1", "roc_auc"} <= metrics.keys()
    assert 0.0 <= metrics["accuracy"] <= 1.0

    predictions = pipeline.predict(sample_df.drop(columns=["churn"]))
    assert len(predictions) == len(sample_df)
