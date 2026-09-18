import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "churn"

NUMERIC_FEATURES = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
]

CATEGORICAL_FEATURES = [
    "region",
    "device_type",
    "payment_method",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(df[NUMERIC_FEATURES].median())
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].fillna("unknown")

    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]
    return X, y


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def class_distribution(y: pd.Series) -> dict:
    return y.value_counts().to_dict()
