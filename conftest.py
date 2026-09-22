import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core import state
from app.ml import model_store, training_history
from app.ml.dataset import load_dataset
from main import app


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 60

    churn = np.array([0] * 30 + [1] * 30)
    rng.shuffle(churn)

    return pd.DataFrame(
        {
            "monthly_fee": rng.uniform(10, 50, n),
            "usage_hours": rng.uniform(0, 300, n),
            "support_requests": rng.integers(0, 10, n),
            "account_age_months": rng.integers(1, 40, n),
            "failed_payments": rng.integers(0, 5, n),
            "region": rng.choice(["america", "europe", "asia"], n),
            "device_type": rng.choice(["mobile", "desktop"], n),
            "payment_method": rng.choice(["card", "paypal"], n),
            "autopay_enabled": rng.integers(0, 2, n),
            "churn": churn,
        }
    )


@pytest.fixture
def sample_client_payload() -> dict:
    return {
        "monthly_fee": 29.9,
        "usage_hours": 120.5,
        "support_requests": 2,
        "account_age_months": 14,
        "failed_payments": 0,
        "region": "europe",
        "device_type": "mobile",
        "payment_method": "card",
        "autopay_enabled": 1,
    }


@pytest.fixture
def api_client(sample_df, tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setattr(state, "dataset_df", sample_df)
    monkeypatch.setattr(state, "_model_record", None)
    monkeypatch.setattr(model_store, "MODEL_PATH", tmp_path / "churn_model.joblib")
    monkeypatch.setattr(
        training_history, "HISTORY_PATH", tmp_path / "training_history.json"
    )
    return TestClient(app)


@pytest.fixture
def real_data_api_client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setattr(state, "dataset_df", load_dataset())
    monkeypatch.setattr(state, "_model_record", None)
    monkeypatch.setattr(model_store, "MODEL_PATH", tmp_path / "churn_model.joblib")
    monkeypatch.setattr(
        training_history, "HISTORY_PATH", tmp_path / "training_history.json"
    )
    return TestClient(app)
