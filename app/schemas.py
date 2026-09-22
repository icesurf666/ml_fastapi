from pydantic import BaseModel


class FeatureVectorChurn(BaseModel):
    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class DatasetRowChurn(FeatureVectorChurn):
    churn: int


class PredictionResponseChurn(BaseModel):
    churn_prediction: int
    churn_probability: float
    not_churn_probability: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "churn_prediction": 0,
                "churn_probability": 0.13,
                "not_churn_probability": 0.87,
            }
        }
    }


class TrainingConfigChurn(BaseModel):
    model_type: str = "logreg"
    hyperparameters: dict = {}


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: dict | list | str | None = None
