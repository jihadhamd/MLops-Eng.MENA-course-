"""Model training pipeline."""

import logging
import pickle
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prodml.config import settings
from prodml.data import clean_data, load_data, split_data
from prodml.features import engineer_features

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


class DurationPredictor:
    """Trains and holds a duration prediction model."""

    def __init__(self) -> None:
        self.model: LinearRegression | None = None
        self.dv: DictVectorizer | None = None

    def train(self, X_train: list[dict[str, Any]], y_train: np.ndarray) -> None:
        """Fit the DictVectorizer and LinearRegression model."""
        self.dv = DictVectorizer()
        X_train_vec = self.dv.fit_transform(X_train)

        self.model = LinearRegression()
        self.model.fit(X_train_vec, y_train)
        logger.info("Model trained successfully")

    def evaluate(
        self, X_val: list[dict[str, Any]], y_val: np.ndarray
    ) -> dict[str, float]:
        """Evaluate on validation data, returning MAE and RMSE."""
        X_val_vec = self.dv.transform(X_val)
        y_pred = self.model.predict(X_val_vec)

        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        return {"mae": mae, "rmse": rmse}

    def save(self, path) -> None:
        """Persist the trained model and vectorizer to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "dv": self.dv}, f)
        logger.info(f"Model saved to {path}")


def main() -> None:
    """Run the full training pipeline end-to-end."""
    df = load_data(str(settings.data_path))
    df = clean_data(df)

    X = engineer_features(df)
    y = df["duration"].values

    X_train, X_val, y_train, y_val = split_data(X, y)

    predictor = DurationPredictor()
    predictor.train(X_train, y_train)

    metrics = predictor.evaluate(X_val, y_val)
    print(f"Validation MAE: {metrics['mae']:.4f}")
    print(f"Validation RMSE: {metrics['rmse']:.4f}")

    predictor.save(settings.model_path)
    print(f"Model saved to {settings.model_path}")


if __name__ == "__main__":
    main()
