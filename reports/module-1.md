# Module 1 Lab Report

## Baseline Metrics (notebooks/00-baseline.ipynb)

- **Validation MAE**: 4.6998
- **Validation RMSE**: 8.4512

Model: LinearRegression with DictVectorizer on `PU_DO` (pickup-dropoff pair) + `trip_distance`, target = trip duration in minutes.

Data: NYC TLC green taxi trip data, May 2026 (`green_tripdata_2026-05.parquet`), 44,921 rows before filtering.

This notebook is deliberately messy — it is the "before" picture, committed as-is.

## Docker Image Size Comparison

| Build type | Size |
|---|---|
| Single-stage | 1.07GB |
| Multi-stage | 1.05GB |

Reduction: ~20MB (~2%)

Note: The reduction here is smaller than typically expected from multi-stage
builds. This is because our dependencies (onnxruntime, scikit-learn, pandas,
pyarrow) install from prebuilt wheels and require no compilation - there's
little "build-only" bloat (compilers, build tools) for the multi-stage
pattern to discard. Multi-stage builds show their biggest benefit when
dependencies must be compiled from source, requiring heavy build toolchains
that aren't needed at runtime. Multi-stage is still used here as the correct
practice (smaller attack surface, cleaner separation of build vs runtime
concerns), even though the size savings are modest for this specific
dependency set.

## Docker Hub

Image: jihadhamy/prodml-api:0.1.0 (also tagged latest)
Verified: pulled fresh (after removing all local images/cache) and confirmed
/health and /predict work correctly with zero local setup - matches the
handbook's acceptance check for Step 7.
