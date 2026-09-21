"""FastAPI service for ride duration prediction."""

import hashlib
import logging
import pickle
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as rt
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from prodml.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    MetadataResponse,
    PredictionRequest,
    PredictionResponse,
)
from prodml.config import settings
from prodml.export import INPUT_NAME, OUTPUT_NAME, export_onnx
from prodml.logging_conf import get_correlation_id, new_correlation_id, setup_logging

setup_logging(level=settings.log_level)
logger = logging.getLogger(__name__)

MODEL_VERSION = "0.1.0"
ONNX_PATH = Path("models/model.onnx")

state: dict = {"session": None, "dv": None, "artifact_hash": None}


def _file_hash(path: Path) -> str:
    """Compute a short SHA256 hash of a file for metadata/traceability."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:12]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model ONCE at startup, not per-request."""
    logger.info("Loading model at startup...")

    if not ONNX_PATH.exists():
        export_onnx(settings.model_path, ONNX_PATH)

    with open(
        settings.model_path, "rb"
    ) as f:  # noqa: ASYNC230 - startup only, runs once
        state["dv"] = pickle.load(f)["dv"]

    state["session"] = rt.InferenceSession(str(ONNX_PATH))
    state["artifact_hash"] = _file_hash(ONNX_PATH)

    logger.info("Model loaded and ready")
    yield
    logger.info("Shutting down")


app = FastAPI(title="Duration Predictor API", version=MODEL_VERSION, lifespan=lifespan)


@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    """Attach a correlation ID to every request, log it, echo it back."""
    cid = new_correlation_id()
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    response = await call_next(request)
    response.headers["X-Request-ID"] = cid
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Clean 422 for bad input, no stack trace leaked."""
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """500 for unexpected errors - logged internally, not leaked to client."""
    logger.error(f"Unexpected error: {exc!r}", exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def _predict(features: dict) -> float:
    """Run one feature dict through the loaded ONNX session."""
    if features.get("trip_distance", 0) > 100:
        logger.warning(
            f"trip_distance outside training range: {features.get('trip_distance')}"
        )
    logger.debug(f"Feature vector: {features}")

    X = state["dv"].transform([features]).toarray().astype(np.float32)
    preds = state["session"].run([OUTPUT_NAME], {INPUT_NAME: X})[0]
    return float(preds.flatten()[0])


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """200 only if the model is actually loaded in memory."""
    loaded = state["session"] is not None
    return HealthResponse(
        status="healthy" if loaded else "unhealthy", model_loaded=loaded
    )


@app.get("/metadata", response_model=MetadataResponse)
async def metadata() -> MetadataResponse:
    """Model version, training date, features, framework, artifact hash."""
    return MetadataResponse(
        model_version=MODEL_VERSION,
        training_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        features=["PU_DO", "trip_distance"],
        framework="scikit-learn + onnxruntime",
        artifact_hash=state["artifact_hash"] or "unknown",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest) -> PredictionResponse:
    """Single prediction."""
    start = time.perf_counter()
    pred = _predict(req.model_dump())
    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Prediction served: {pred:.2f} minutes")
    return PredictionResponse(
        prediction=pred,
        model_version=MODEL_VERSION,
        correlation_id=get_correlation_id(),
        latency_ms=latency_ms,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(req: BatchPredictionRequest) -> BatchPredictionResponse:
    """Batch prediction - list in, list out."""
    start = time.perf_counter()
    preds = [_predict(p.model_dump()) for p in req.predictions]
    latency_ms = (time.perf_counter() - start) * 1000

    return BatchPredictionResponse(
        predictions=preds,
        correlation_id=get_correlation_id(),
        latency_ms=latency_ms,
    )
