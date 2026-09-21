"""Tests for ONNX export and pickle/ONNX parity."""

import numpy as np
import pytest

from prodml.export import export_onnx, predict_onnx, predict_pickle


@pytest.fixture
def onnx_path(trained_model, tmp_path):
    """Export a fresh ONNX model into a temp directory for isolated testing."""
    path = tmp_path / "model.onnx"
    export_onnx(trained_model, path)
    return path


class TestExportOnnx:
    def test_export_creates_file(self, onnx_path):
        assert onnx_path.exists()

    def test_export_creates_feature_names_file(self, onnx_path):
        meta_path = onnx_path.with_suffix(".feature_names.json")
        assert meta_path.exists()


class TestParity:
    def test_pickle_onnx_parity(self, trained_model, onnx_path):
        """Pickle and ONNX predictions must match within atol=1e-4."""
        import pickle

        with open(trained_model, "rb") as f:
            dv = pickle.load(f)["dv"]

        test_rows = [
            {"PU_DO": "74_41", "trip_distance": 5.0},
            {"PU_DO": "42_42", "trip_distance": 10.0},
            {"PU_DO": "1_1", "trip_distance": 1.0},
        ]
        X = dv.transform(test_rows).toarray().astype(np.float32)

        y_pkl = predict_pickle(trained_model, X)
        y_onnx = predict_onnx(onnx_path, X)

        assert np.allclose(y_pkl, y_onnx, atol=1e-4)


class TestValidation:
    """Uses monkeypatch to simulate a mismatched ONNX file without
    needing a real broken model on disk."""

    def test_predict_onnx_rejects_wrong_input_name(self, onnx_path, monkeypatch):
        import onnxruntime as rt

        import prodml.export as export_module

        real_session_cls = rt.InferenceSession

        class FakeInput:
            name = "wrong_input_name"
            shape = (None, 100)

        class FakeSession:
            def __init__(self, *args, **kwargs):
                self._real = real_session_cls(*args, **kwargs)

            def get_inputs(self):
                return [FakeInput()]

            def get_outputs(self):
                return self._real.get_outputs()

        monkeypatch.setattr(export_module.rt, "InferenceSession", FakeSession)

        X = np.zeros((1, 100), dtype=np.float32)
        with pytest.raises(ValueError, match="input tensor name mismatch"):
            predict_onnx(onnx_path, X)
