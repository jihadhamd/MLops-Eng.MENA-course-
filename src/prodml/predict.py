"""Prediction logic — load a trained model and serve predictions."""

import logging
import pickle
import time
from functools import wraps
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def timed(func):
    """Decorator that logs and returns execution time in milliseconds."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"{func.__name__} took {elapsed_ms:.2f}ms")
        return result, elapsed_ms

    return wrapper


class DurationPredictor:
    """Loads a trained model and serves predictions."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.model = None
        self.dv = None
        self.load()

    def load(self) -> None:
        """Load the pickled model and vectorizer from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        with open(self.model_path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.dv = data["dv"]
        logger.info(f"Model loaded from {self.model_path}")

    @timed
    def predict_one(self, features: dict[str, Any]) -> float:
        """Predict duration for a single set of features."""
        X = self.dv.transform([features])
        pred = self.model.predict(X)[0]
        return float(pred)

    def predict_batch(self, features: list[dict[str, Any]]) -> list[float]:
        """Predict duration for a batch of feature dicts."""
        X = self.dv.transform(features)
        preds = self.model.predict(X)
        return preds.tolist()
