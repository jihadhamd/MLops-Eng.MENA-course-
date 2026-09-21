"""Tests for prediction logic (DurationPredictor)."""


class TestPredictOne:
    """Single prediction behaviour."""

    def test_returns_float(self, predictor, sample_features):
        pred, _ = predictor.predict_one(sample_features)
        assert isinstance(pred, float)

    def test_sane_range(self, predictor, sample_features):
        """A 5-mile trip should predict somewhere between 0 and 3 hours."""
        pred, _ = predictor.predict_one(sample_features)
        assert 0 < pred < 180

    def test_deterministic(self, predictor, sample_features):
        """Same input twice must give the exact same prediction."""
        pred1, _ = predictor.predict_one(sample_features)
        pred2, _ = predictor.predict_one(sample_features)
        assert pred1 == pred2

    def test_returns_latency(self, predictor, sample_features):
        _, latency_ms = predictor.predict_one(sample_features)
        assert isinstance(latency_ms, float)
        assert latency_ms >= 0


class TestPredictBatch:
    """Batch prediction behaviour."""

    def test_batch_returns_list_of_floats(self, predictor):
        features = [
            {"PU_DO": "74_41", "trip_distance": 5.0},
            {"PU_DO": "42_42", "trip_distance": 10.0},
        ]
        preds = predictor.predict_batch(features)

        assert len(preds) == 2
        assert all(isinstance(p, float) for p in preds)

    def test_batch_matches_single_predictions(self, predictor):
        """Batch predictions should equal calling predict_one individually."""
        features = [{"PU_DO": "74_41", "trip_distance": 5.0}]
        batch_preds = predictor.predict_batch(features)
        single_pred, _ = predictor.predict_one(features[0])

        assert abs(batch_preds[0] - single_pred) < 1e-6
