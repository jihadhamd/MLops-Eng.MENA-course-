# Module 1 Lab Report

## Baseline Metrics (notebooks/00-baseline.ipynb)

- **Validation MAE**: 4.6998
- **Validation RMSE**: 8.4512

Model: LinearRegression with DictVectorizer on `PU_DO` (pickup-dropoff pair) + `trip_distance`, target = trip duration in minutes.

Data: NYC TLC green taxi trip data, May 2026 (`green_tripdata_2026-05.parquet`), 44,921 rows before filtering.

This notebook is deliberately messy — it is the "before" picture, committed as-is.
