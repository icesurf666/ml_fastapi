# ML Churn Service

A FastAPI service for predicting customer churn: trains a classification model on tabular customer data and serves predictions through a REST API.

## Project structure

```
main.py                  # entrypoint, creates the FastAPI app and includes routers
app/
  api/                    # HTTP layer: routers and /docs examples
    predict.py            # POST /predict
    dataset.py             # GET /dataset/preview, /info, /split-info
    model.py                # POST /model/train, GET /model/schema, /metrics, /status
    health.py                # GET /health
    openapi_examples.py       # request/response examples for Swagger UI
  ml/                     # ML pipeline
    dataset.py             # reads churn_dataset.csv
    preprocessing.py       # feature selection, missing values, train/test split
    model.py                # model selection, sklearn Pipeline, training, metrics
    model_store.py          # save/load the trained model (joblib)
    training_history.py     # training run history (JSON)
  core/                   # infrastructure
    state.py                # app state (dataset, current model)
    logging_config.py        # logging setup
    error_handlers.py        # unified error format
  schemas.py              # Pydantic request/response models
data/
  churn_dataset.csv       # training dataset
models/                   # trained model and training history (created on first /model/train, not in git)
tests/                    # pytest: unit and integration tests
```

## Dataset `churn_dataset.csv`

2000 rows, 10 columns. Each row is one customer.

| Column | Type | Description |
|---|---|---|
| `monthly_fee` | float | monthly subscription fee |
| `usage_hours` | float | hours of service usage |
| `support_requests` | int | number of support tickets |
| `account_age_months` | int | account age in months |
| `failed_payments` | int | number of failed payments |
| `region` | str | customer region (categorical) |
| `device_type` | str | device type (categorical) |
| `payment_method` | str | payment method (categorical) |
| `autopay_enabled` | int | whether autopay is enabled (0/1) |
| `churn` | int | target: customer churned (1) or not (0) |

The current feature list and types are also available via `GET /model/schema`.

## Running locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

uvicorn main:app --reload
```

The service starts at `http://127.0.0.1:8000`, docs at `http://127.0.0.1:8000/docs`.

## Running in Docker

```bash
docker build -t churn-service .
docker run -d -p 8000:8000 churn-service
```

Check it's up:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/docs
```

The container starts without a trained model (`models/` is not copied into the image) — train one via `/model/train` after startup.

## Tests

```bash
source venv/bin/activate
pytest
```

## Example requests

### Train a model — `POST /model/train`

```bash
curl -X POST http://127.0.0.1:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "random_forest", "hyperparameters": {"n_estimators": 100, "max_depth": 5}}'
```

The request body is optional — without it, a `LogisticRegression` with default settings is trained. Available `model_type` values: `logreg`, `random_forest`.

Response:
```json
{"accuracy": 0.79, "f1": 0.32, "roc_auc": 0.68}
```

### Get a prediction — `POST /predict`

Single client:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "monthly_fee": 29.9,
    "usage_hours": 120.5,
    "support_requests": 2,
    "account_age_months": 14,
    "failed_payments": 0,
    "region": "europe",
    "device_type": "mobile",
    "payment_method": "card",
    "autopay_enabled": 1
  }'
```

Response:
```json
{"churn_prediction": 0, "churn_probability": 0.13, "not_churn_probability": 0.87}
```

For multiple clients, send a JSON array of objects instead of a single object — the response will be an array of predictions in the same order.

If the model hasn't been trained yet, `/predict` returns `400` with a body like `{"code": "bad_request", "message": "...", "details": null}`.

## Other endpoints

- `GET /health` — dataset and trained model availability
- `GET /model/status` — whether a model is trained, when, and with what metrics
- `GET /model/metrics?limit=5&model_type=logreg` — training history
- `GET /dataset/preview`, `/dataset/info`, `/dataset/split-info` — inspect the dataset
