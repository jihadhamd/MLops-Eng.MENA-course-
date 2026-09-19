# MLOps Practitioner — Mini Project 1

NYC taxi ride duration prediction service. Notebook to production-ready Python package.

## Workflow

pip install -e ".[dev]"
ruff check src tests && black --check src tests
pytest -v --cov=src/prodml --cov-report=term-missing
python -m prodml.train
uvicorn prodml.api.main:app --reload --port 8000

## Baseline

- Validation MAE: 4.6998
- Validation RMSE: 8.4512
- Model: LinearRegression + DictVectorizer on PU_DO (pickup-dropoff pair) and trip_distance
- Data: NYC TLC green taxi trip data, May 2026

## Repository Structure

notebooks/00-baseline.ipynb   - messy "before" notebook
src/prodml/
  config.py                   - settings via environment variables
  data.py                     - load, clean, split data
  features.py                 - PU_DO feature engineering
  train.py                    - DurationPredictor training and evaluation
  predict.py                  - DurationPredictor loading and prediction
  api/                        - FastAPI service (Step 5)
tests/                        - pytest suite (Step 6)
docker/                       - Dockerfile and compose (Step 7)
models/                       - baseline.pkl, model.pkl
reports/module-1.md           - metrics and self-assessment
