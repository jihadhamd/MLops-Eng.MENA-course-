# MLOps Practitioner — Mini Project 1

NYC taxi ride duration prediction service. From baseline notebook to a tested, containerized, production-ready FastAPI service.

## Quick Start (3 commands)

pip install -e ".[dev]"
python -m prodml.train
uvicorn prodml.api.main:app --reload --port 8000

## Or run it in Docker (zero setup)

docker run -p 8000:8000 jihadhamy/prodml-api:0.1.0

## Example request

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"PU_DO": "74_41", "trip_distance": 5.0}'

Response:
{"prediction": 19.58, "model_version": "0.1.0", "correlation_id": "...", "latency_ms": 2.38}

## Workflow

pip install -e ".[dev]"                                # install
ruff check src tests && black --check src tests        # lint
pytest -v --cov=src/prodml --cov-report=term-missing   # test
python -m prodml.train                                 # train
uvicorn prodml.api.main:app --reload --port 8000        # serve

## Repository Structure

notebooks/00-baseline.ipynb   - messy "before" notebook (MAE 4.6998, RMSE 8.4512)
src/prodml/
  config.py                   - settings via environment variables
  data.py                     - load, clean, split data
  features.py                 - PU_DO feature engineering + validation
  train.py                    - DurationPredictor training and evaluation
  predict.py                  - DurationPredictor loading and prediction, @timed decorator
  export.py                   - ONNX export, named tensors, input/output validation
  logging_conf.py             - structured JSON logging, correlation IDs
  api/
    main.py                   - FastAPI app, 4 endpoints, startup model loading
    schemas.py                - Pydantic request/response models
tests/                        - pytest suite, 75%+ coverage
  conftest.py                 - shared fixtures
  test_features.py            - edge cases, parametrized boundaries
  test_predict.py             - prediction correctness, determinism
  test_api.py                 - endpoint behaviour, validation, schema
  test_serialization.py       - pickle/ONNX parity, monkeypatch validation test
docker/
  Dockerfile                  - multi-stage build, non-root user, healthcheck
  Dockerfile.singlestage       - comparison-only, not used in production
scripts/
  benchmark_serialization.py  - ONNX export + parity + latency benchmark
models/                       - baseline.pkl, model.pkl, model.onnx
reports/
  module-1.md                 - metrics, format comparison, Docker comparison, self-assessment
  serialization-formats.md    - format comparison table

## Endpoints

GET  /health         - 200 only if model is loaded in memory
GET  /metadata        - model version, features, framework, artifact hash
POST /predict          - single prediction
POST /predict/batch    - batch prediction

## Docker Hub

Image: jihadhamy/prodml-api:0.1.0 (also tagged latest)

## Key Metrics

- Validation MAE: 4.6998 | RMSE: 8.4512
- Test coverage: 75.46%
- Pickle vs ONNX parity: max diff 4.17e-6 (passes atol=1e-4)
- Docker image size: multi-stage 1.05GB vs single-stage 1.07GB

## Known Optimization Opportunity (not yet implemented)

The serving path currently unpickles a scikit-learn DictVectorizer at startup
to build the ONNX input vector, which is why scikit-learn (and its
dependencies) are still required in the runtime image even though the
model itself runs on ONNX. A future optimization would replace this with a
lightweight, dependency-free vectorization function built from the saved
model.feature_names.json column order, removing the scikit-learn/pandas
dependency from the serving image entirely and reducing its size
significantly.
