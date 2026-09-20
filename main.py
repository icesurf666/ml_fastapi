from fastapi import FastAPI, HTTPException

from dataset import load_dataset
from model import train_churn_model
from model_store import load_churn_model, save_churn_model
from preprocessing import class_distribution, prepare_features, split_dataset
from schemas import FeatureVectorChurn

app = FastAPI()
dataset_df = load_dataset()
model_record = load_churn_model()


@app.get("/")
def read_root():
    return {"message": "ml churn service is running"}


@app.post("/predict")
def predict(features: FeatureVectorChurn):
    return features


@app.get("/dataset/preview")
def dataset_preview(n: int = 5):
    return dataset_df.head(n).to_dict(orient="records")


@app.get("/dataset/info")
def dataset_info():
    return {
        "rows": dataset_df.shape[0],
        "columns": dataset_df.shape[1],
        "feature_names": dataset_df.columns.tolist(),
        "churn_distribution": dataset_df["churn"].value_counts().to_dict(),
    }


@app.get("/dataset/split-info")
def dataset_split_info():
    X, y = prepare_features(dataset_df)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    return {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_churn_distribution": class_distribution(y_train),
        "test_churn_distribution": class_distribution(y_test),
    }


@app.post("/model/train")
def model_train():
    global model_record

    if dataset_df is None or dataset_df.empty:
        raise HTTPException(status_code=400, detail="Dataset is not loaded or empty")

    pipeline, metrics = train_churn_model(dataset_df)
    model_record = save_churn_model(pipeline, metrics)
    return metrics


@app.get("/model/status")
def model_status():
    if model_record is None:
        return {"trained": False, "trained_at": None, "metrics": None}

    return {
        "trained": True,
        "trained_at": model_record["trained_at"],
        "metrics": model_record["metrics"],
    }
