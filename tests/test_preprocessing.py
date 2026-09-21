from preprocessing import (
    ALL_FEATURES,
    class_distribution,
    prepare_features,
    select_and_order_features,
    split_dataset,
)


def test_select_and_order_features_returns_all_features_in_order(sample_df):
    X = select_and_order_features(sample_df)

    assert list(X.columns) == ALL_FEATURES


def test_select_and_order_features_fills_missing_numeric_with_median(sample_df):
    df = sample_df.copy()
    df.loc[0, "monthly_fee"] = None

    X = select_and_order_features(df)

    assert not X["monthly_fee"].isna().any()


def test_select_and_order_features_fills_missing_categorical_with_unknown(sample_df):
    df = sample_df.copy()
    df.loc[0, "region"] = None

    X = select_and_order_features(df)

    assert X.loc[0, "region"] == "unknown"


def test_prepare_features_splits_into_X_and_y(sample_df):
    X, y = prepare_features(sample_df)

    assert list(X.columns) == ALL_FEATURES
    assert y.name == "churn"
    assert len(X) == len(y) == len(sample_df)


def test_split_dataset_keeps_both_classes_in_train_and_test(sample_df):
    X, y = prepare_features(sample_df)
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2)

    assert len(X_train) + len(X_test) == len(sample_df)
    assert set(class_distribution(y_train).keys()) == {0, 1}
    assert set(class_distribution(y_test).keys()) == {0, 1}
