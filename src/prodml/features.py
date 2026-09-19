"""Feature engineering for the duration prediction model."""

from typing import Any

import pandas as pd


def engineer_features(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build the PU_DO feature and select model inputs."""
    df = df.copy()
    df["PU_DO"] = df["PULocationID"].astype(str) + "_" + df["DOLocationID"].astype(str)

    features = df[["PU_DO", "trip_distance"]].to_dict("records")
    return features


def validate_features(features: dict[str, Any]) -> bool:
    """Check that a single feature dict has sane values."""
    distance = features.get("trip_distance", 0)
    if distance <= 0 or distance > 200:
        return False
    return bool(features.get("PU_DO"))
