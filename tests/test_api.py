def test_model_status_is_not_trained_initially(api_client):
    response = api_client.get("/model/status")

    assert response.status_code == 200
    assert response.json()["trained"] is False


def test_predict_without_trained_model_returns_clean_400(api_client, sample_client_payload):
    response = api_client.post("/predict", json=sample_client_payload)

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "bad_request"
    assert "not trained" in body["message"].lower()


def test_predict_missing_fields_returns_422(api_client):
    response = api_client.post("/predict", json={"monthly_fee": 29.9})

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_full_train_status_predict_flow(api_client, sample_client_payload):
    train_response = api_client.post("/model/train", json={})
    assert train_response.status_code == 200
    metrics = train_response.json()
    assert {"accuracy", "f1", "roc_auc"} <= metrics.keys()

    status_response = api_client.get("/model/status")
    assert status_response.status_code == 200
    status = status_response.json()
    assert status["trained"] is True
    assert status["model_type"] == "logreg"

    predict_response = api_client.post("/predict", json=sample_client_payload)
    assert predict_response.status_code == 200
    prediction = predict_response.json()
    assert prediction["churn_prediction"] in (0, 1)
    assert 0.0 <= prediction["churn_probability"] <= 1.0
    assert 0.0 <= prediction["not_churn_probability"] <= 1.0


def test_predict_batch_returns_list_of_same_length(api_client, sample_client_payload):
    api_client.post("/model/train", json={})

    response = api_client.post("/predict", json=[sample_client_payload, sample_client_payload])

    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    assert len(results) == 2


def test_full_flow_reads_real_dataset_trains_and_predicts(
    real_data_api_client, sample_client_payload
):
    train_response = real_data_api_client.post("/model/train", json={})
    assert train_response.status_code == 200
    metrics = train_response.json()
    assert {"accuracy", "f1", "roc_auc"} <= metrics.keys()

    status_response = real_data_api_client.get("/model/status")
    assert status_response.status_code == 200
    assert status_response.json()["trained"] is True

    predict_response = real_data_api_client.post("/predict", json=sample_client_payload)
    assert predict_response.status_code == 200
    prediction = predict_response.json()
    assert prediction["churn_prediction"] in (0, 1)
