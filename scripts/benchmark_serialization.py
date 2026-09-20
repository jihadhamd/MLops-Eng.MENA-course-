"""Export to ONNX, verify parity with pickle, and benchmark latency.

Run: python scripts/benchmark_serialization.py
"""

import pickle
import time
from pathlib import Path

import numpy as np

from prodml.config import settings
from prodml.data import load_data, clean_data, split_data
from prodml.features import engineer_features
from prodml.export import export_onnx, predict_pickle, predict_onnx


def main() -> None:
    pickle_path = settings.model_path
    onnx_path = Path("models/model.onnx")

    # 1. Export
    print("Exporting ONNX model...")
    export_onnx(pickle_path, onnx_path)

    # 2. Build 500 real validation rows (same pipeline as training)
    df = load_data(str(settings.data_path))
    df = clean_data(df)
    X_dicts = engineer_features(df)
    y = df["duration"].values
    _, X_val_dicts, _, _ = split_data(X_dicts, y)

    X_val_dicts = X_val_dicts[:500]
    print(f"Using {len(X_val_dicts)} validation rows for parity + benchmark")

    with open(pickle_path, "rb") as f:
        dv = pickle.load(f)["dv"]
    X_val = dv.transform(X_val_dicts).toarray().astype(np.float32)

    # 3. Parity test
    print("\nRunning parity test...")
    y_pkl = predict_pickle(pickle_path, X_val)
    y_onnx = predict_onnx(onnx_path, X_val)

    max_diff = np.max(np.abs(y_pkl - y_onnx))
    passed = np.allclose(y_pkl, y_onnx, atol=1e-4)
    print(f"Max absolute difference: {max_diff:.8f}")
    print(f"Parity test (atol=1e-4): {'PASSED' if passed else 'FAILED'}")

    if not passed:
        raise SystemExit("Parity test failed - pickle and ONNX predictions diverge")

    # 4. Latency benchmark
    print("\nBenchmarking latency (10 runs each)...")

    pkl_times = []
    for _ in range(10):
        start = time.perf_counter()
        predict_pickle(pickle_path, X_val)
        pkl_times.append((time.perf_counter() - start) * 1000)

    onnx_times = []
    for _ in range(10):
        start = time.perf_counter()
        predict_onnx(onnx_path, X_val)
        onnx_times.append((time.perf_counter() - start) * 1000)

    print(
        f"\nPickle - mean: {np.mean(pkl_times):.3f}ms, p95: {np.percentile(pkl_times, 95):.3f}ms"
    )
    print(
        f"ONNX   - mean: {np.mean(onnx_times):.3f}ms, p95: {np.percentile(onnx_times, 95):.3f}ms"
    )


if __name__ == "__main__":
    main()
