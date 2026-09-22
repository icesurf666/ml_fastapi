from fastapi import APIRouter

from app.core import state
from app.ml.preprocessing import class_distribution, prepare_features, split_dataset

router = APIRouter(prefix="/dataset")


@router.get("/preview")
def dataset_preview(n: int = 5):
    return state.dataset_df.head(n).to_dict(orient="records")


@router.get("/info")
def dataset_info():
    df = state.dataset_df
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "feature_names": df.columns.tolist(),
        "churn_distribution": df["churn"].value_counts().to_dict(),
    }


@router.get("/split-info")
def dataset_split_info():
    X, y = prepare_features(state.dataset_df)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    return {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_churn_distribution": class_distribution(y_train),
        "test_churn_distribution": class_distribution(y_test),
    }
