"""Shared pytest fixtures."""

import pytest
from fastapi.testclient import TestClient

from prodml.config import settings
from prodml.predict import DurationPredictor


@pytest.fixture(scope="session")
def trained_model():
    """Ensure the model artifact exists before running tests that need it."""
    if not settings.model_path.exists():
        pytest.skip("Model not found - run: python -m prodml.train")
    return settings.model_path


@pytest.fixture
def sample_features():
    """A valid feature dict for a single prediction."""
    return {"PU_DO": "74_41", "trip_distance": 5.0}


@pytest.fixture
def predictor(trained_model):
    """A loaded DurationPredictor instance."""
    return DurationPredictor(trained_model)


@pytest.fixture
def client():
    """FastAPI TestClient - triggers the app's lifespan (model loading)."""
    from prodml.api.main import app

    with TestClient(app) as c:
        yield c
