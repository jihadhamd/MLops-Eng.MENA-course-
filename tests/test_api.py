"""Tests for the FastAPI service endpoints."""


class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_reports_model_loaded(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


class TestMetadata:
    def test_metadata_returns_200(self, client):
        response = client.get("/metadata")
        assert response.status_code == 200

    def test_metadata_schema(self, client):
        data = client.get("/metadata").json()
        for field in [
            "model_version",
            "training_date",
            "features",
            "framework",
            "artifact_hash",
        ]:
            assert field in data


class TestPredict:
    def test_predict_happy_path(self, client, sample_features):
        response = client.post("/predict", json=sample_features)
        assert response.status_code == 200

    def test_predict_response_schema(self, client, sample_features):
        data = client.post("/predict", json=sample_features).json()
        for field in ["prediction", "model_version", "correlation_id", "latency_ms"]:
            assert field in data
        assert isinstance(data["prediction"], float)

    def test_predict_invalid_negative_distance_returns_422(self, client):
        response = client.post(
            "/predict", json={"PU_DO": "74_41", "trip_distance": -5.0}
        )
        assert response.status_code == 422

    def test_predict_invalid_missing_field_returns_422(self, client):
        response = client.post("/predict", json={"PU_DO": "74_41"})
        assert response.status_code == 422

    def test_predict_correlation_id_in_response_header(self, client, sample_features):
        response = client.post("/predict", json=sample_features)
        assert "x-request-id" in response.headers


class TestPredictBatch:
    def test_batch_happy_path(self, client, sample_features):
        response = client.post(
            "/predict/batch", json={"predictions": [sample_features, sample_features]}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 2
