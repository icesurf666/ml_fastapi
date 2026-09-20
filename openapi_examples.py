from schemas import ErrorResponse

PREDICT_BODY_EXAMPLES = {
    "single_client": {
        "summary": "Single client",
        "description": "Predict churn for one client",
        "value": {
            "monthly_fee": 29.9,
            "usage_hours": 120.5,
            "support_requests": 2,
            "account_age_months": 14,
            "failed_payments": 0,
            "region": "europe",
            "device_type": "mobile",
            "payment_method": "card",
            "autopay_enabled": 1,
        },
    },
    "batch_clients": {
        "summary": "Multiple clients",
        "description": "Predict churn for a list of clients",
        "value": [
            {
                "monthly_fee": 9.99,
                "usage_hours": 5.0,
                "support_requests": 8,
                "account_age_months": 1,
                "failed_payments": 3,
                "region": "asia",
                "device_type": "desktop",
                "payment_method": "paypal",
                "autopay_enabled": 0,
            },
            {
                "monthly_fee": 49.99,
                "usage_hours": 300.0,
                "support_requests": 0,
                "account_age_months": 36,
                "region": "europe",
                "device_type": "mobile",
                "payment_method": "card",
                "failed_payments": 0,
                "autopay_enabled": 1,
            },
        ],
    },
}

PREDICT_ERROR_RESPONSES = {
    400: {
        "model": ErrorResponse,
        "description": "Model is not trained yet",
        "content": {
            "application/json": {
                "example": {
                    "code": "bad_request",
                    "message": "Model is not trained yet. Call POST /model/train first.",
                    "details": None,
                }
            }
        },
    },
    422: {
        "model": ErrorResponse,
        "description": "Invalid client data: wrong number of features or wrong types",
        "content": {
            "application/json": {
                "example": {
                    "code": "validation_error",
                    "message": "Invalid request data",
                    "details": [
                        {
                            "type": "missing",
                            "loc": ["body", "monthly_fee"],
                            "msg": "Field required",
                        },
                        {
                            "type": "float_parsing",
                            "loc": ["body", "usage_hours"],
                            "msg": "Input should be a valid number",
                        },
                    ],
                }
            }
        },
    },
}

TRAIN_ERROR_RESPONSES = {
    400: {
        "model": ErrorResponse,
        "description": "Empty dataset or unknown model_type/hyperparameters",
        "content": {
            "application/json": {
                "example": {
                    "code": "bad_request",
                    "message": "Unknown model_type 'xgboost'. Available: ['logreg', 'random_forest']",
                    "details": None,
                }
            }
        },
    },
}
