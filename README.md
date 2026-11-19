# autonomous-soc-self-healing
ML &amp; Risk scoring pipeline for Autonomous SOC &amp; Self-Healing Network.
# Autonomous SOC & Self-Healing Network

Graduation project focused on detecting and classifying network anomalies and passing actionable alerts to the Controller layer.

## What's Here So Far

- **`features/pipeline.py`**: Feature preparation (`prepare_soc_features`) and the sequential ML pipeline (`run_sequential_pipeline`):
  - Normalizes common telemetry column names and separates flow context (`Source IP`, `Destination IP`, `Source Port`, `Destination Port`, `Protocol`, `Timestamp`) from numeric features.
  - Sorts flows chronologically when timestamps are available, adds rolling window statistics (`mean` and `std`), replaces `inf`/`-inf` with `NaN`, and imputes missing values using column medians (with `0.0` fallback).
  - Runs **`IsolationForest`** first to flag anomalous flows and compute `anomaly_score`.
  - Trains and evaluates a **`RandomForestClassifier`** (with `StandardScaler`) when multi-class labels are provided, reporting `macro_f1`, `classification_report`, `top_features`, and per-anomaly `attack_class` + `confidence`.

## Setup & Usage

Requires Python 3.11 with `numpy`, `pandas`, and `scikit-learn`:

```bash
pip install numpy pandas scikit-learn
python features/pipeline.py
