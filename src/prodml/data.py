"""Data loading and train/validation splitting."""

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    """Load taxi trip data from a Parquet file."""
    df = pd.read_parquet(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Compute duration and filter out bad rows."""
    df = df.copy()
    df["duration"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 200)]
    df = df[(df["duration"] > 0) & (df["duration"] < 180)]

    return df


def split_data(
    X: list[dict[str, Any]],
    y: Any,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split features and target into train/validation sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
