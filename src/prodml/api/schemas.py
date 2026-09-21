"""Pydantic request/response models for the API."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Single prediction request."""

    PU_DO: str = Field(..., description="Pickup-Dropoff location pair, e.g. '74_41'")
    trip_distance: float = Field(
        ..., gt=0, lt=200, description="Trip distance in miles"
    )

    model_config = {
        "json_schema_extra": {"example": {"PU_DO": "74_41", "trip_distance": 5.0}}
    }


class PredictionResponse(BaseModel):
    """Single prediction response."""

    prediction: float
    model_version: str
    correlation_id: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    predictions: list[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: list[float]
    correlation_id: str
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool


class MetadataResponse(BaseModel):
    """Model metadata response."""

    model_version: str
    training_date: str
    features: list[str]
    framework: str
    artifact_hash: str
