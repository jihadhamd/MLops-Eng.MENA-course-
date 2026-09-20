"""Model export to ONNX and cross-format prediction helpers.

Input tensor:  "duration_features", dtype float32, shape [batch, n_features]
               (vectorized PU_DO one-hot columns + trip_distance, in the
               exact column order saved to <name>.feature_names.json)
Output tensor: "duration_minutes", dtype float32, shape [batch, 1]
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

logger = logging.getLogger(__name__)

INPUT_NAME = "duration_features"
OUTPUT_NAME = "duration_minutes"


def export_onnx(pickle_path: Path, onnx_path: Path) -> None:
    """Convert a pickled sklearn model + vectorizer into an ONNX file."""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    model = data["model"]
    dv = data["dv"]
    feature_names = dv.get_feature_names_out().tolist()
    n_features = len(feature_names)

    initial_type = [(INPUT_NAME, FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        final_types=[(OUTPUT_NAME, FloatTensorType([None, 1]))],
    )

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    meta_path = onnx_path.with_suffix(".feature_names.json")
    with open(meta_path, "w") as f:
        json.dump(feature_names, f, indent=2)

    logger.info(f"ONNX model exported to {onnx_path}")
    logger.info(f"Feature order ({n_features} columns) saved to {meta_path}")


def _validate_input(
    X: np.ndarray, expected_n_features: int | None = None
) -> np.ndarray:
    """Validate and coerce input array before inference.

    Raises ValueError with a clear message on any mismatch instead of
    letting sklearn/onnxruntime fail with a cryptic internal error.
    """
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError(
            f"Expected a 2D array of shape [batch, n_features], got shape {X.shape}"
        )

    if expected_n_features is not None and X.shape[1] != expected_n_features:
        raise ValueError(
            f"Feature count mismatch: model expects {expected_n_features} "
            f"features, got {X.shape[1]}"
        )

    if not np.issubdtype(X.dtype, np.floating):
        raise ValueError(f"Expected a float array, got dtype {X.dtype}")

    return X.astype(np.float32)


def _validate_output(preds: np.ndarray, expected_batch_size: int) -> np.ndarray:
    """Validate model output shape before returning it."""
    preds = np.asarray(preds).flatten()

    if preds.shape[0] != expected_batch_size:
        raise ValueError(
            f"Output batch size mismatch: expected {expected_batch_size} "
            f"predictions, got {preds.shape[0]}"
        )

    if not np.issubdtype(preds.dtype, np.floating):
        raise ValueError(f"Expected float predictions, got dtype {preds.dtype}")

    return preds


def predict_pickle(pickle_path: Path, X: np.ndarray) -> np.ndarray:
    """Predict using the pickled sklearn model directly, with validation."""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    model = data["model"]
    n_features = model.coef_.shape[0] if hasattr(model, "coef_") else None

    X = _validate_input(X, expected_n_features=n_features)
    preds = model.predict(X)
    return _validate_output(preds, expected_batch_size=X.shape[0])


def predict_onnx(onnx_path: Path, X: np.ndarray) -> np.ndarray:
    """Predict using the ONNX runtime session, with validation.

    Confirms the array's feature dimension matches the model's declared
    input shape, and that the input/output tensor names in the file
    match the expected INPUT_NAME / OUTPUT_NAME constants.
    """
    sess = rt.InferenceSession(str(onnx_path))

    session_input = sess.get_inputs()[0]
    session_output = sess.get_outputs()[0]

    if session_input.name != INPUT_NAME:
        raise ValueError(
            f"ONNX input tensor name mismatch: expected '{INPUT_NAME}', "
            f"file has '{session_input.name}'"
        )
    if session_output.name != OUTPUT_NAME:
        raise ValueError(
            f"ONNX output tensor name mismatch: expected '{OUTPUT_NAME}', "
            f"file has '{session_output.name}'"
        )

    expected_n_features = session_input.shape[1]
    X = _validate_input(X, expected_n_features=expected_n_features)

    preds = sess.run([OUTPUT_NAME], {INPUT_NAME: X})[0]
    return _validate_output(preds, expected_batch_size=X.shape[0])
