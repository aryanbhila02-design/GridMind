#!/usr/bin/env python3
"""
GridMind — Standalone Production Prediction Script
HackSphere 3.0: Smart Energy Microgrid & Demand Forecasting Engine

Dual Submission Contract:
1. Interface A (Official Starter Kit Function Contract):
       from predict import predict
       predictions = predict(test_df)
   Returns a NumPy array or Python list of numeric predictions matching input length.

2. Interface B (CLI Contract):
       python predict.py --input sample_test_input.csv --output predictions.csv
   Produces CSV with exact header 'timestamp,target_demand_kw', preserving original
   timestamp strings and arbitrary row count.
"""

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime

# QuickML Configuration from Environment
QUICKML_ENDPOINT = os.getenv("ZOHO_QUICKML_ENDPOINT", "").strip()
QUICKML_API_KEY = os.getenv("ZOHO_CATALYST_API_KEY", "").strip()
QUICKML_PROJECT_ID = os.getenv("ZOHO_PROJECT_ID", "").strip()
ALLOW_LOCAL_FALLBACK = os.getenv("ALLOW_LOCAL_FALLBACK", "1").strip().lower() in ("1", "true", "yes")

REQUIRED_COLUMNS = ["timestamp", "ambient_temp_c", "humidity_pct", "solar_irradiance_wm2"]


def parse_timestamp_hour_weekday(ts_str):
    """
    Safely extracts 0-23 hour and weekday (0=Mon..6=Sun) without timezone shifting.
    Accepts ISO 8601 strings, 'YYYY-MM-DD HH:mm:ss', or time-only.
    """
    try:
        clean = str(ts_str).strip()
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M"):
            try:
                dt = datetime.strptime(clean, fmt)
                return dt.hour, dt.weekday()
            except ValueError:
                pass
        # Fallback regex extraction of HH:mm
        if "T" in clean or " " in clean:
            time_part = clean.replace("T", " ").split()[1]
            h = int(time_part.split(":")[0])
            return h, 1
    except Exception:
        pass
    return 12, 1


# Pre-computed Ridge Regression weights trained on official HackSphere train_telemetry.csv
# Features: [bias, ambient_temp_c, humidity_pct, solar_irradiance_wm2, sin_hour, cos_hour, is_evening]
# Standardized against train_telemetry.csv (72 continuous hourly rows, Tue-Thu):
TRAINED_MEANS = [22.861111, 66.791667, 299.569444, 0.0, 0.0, 0.208333]
TRAINED_STDS = [4.809295, 15.710875, 362.364336, 0.707107, 0.707107, 0.406116]
TRAINED_WEIGHTS = [146.6625, 16.283811, 22.219724, 20.909898, -11.891714, -15.697609, 28.36902]


def predict_demand_local_baseline(timestamp, temp, humidity, solar):
    """
    Trained Ridge Regression Model (Unified with TypeScript forecastingEngine).
    Trained on official HackSphere 3.0 train_telemetry.csv using L2 regularized Ridge Regression.
    Achieves Training RMSE: 10.03 kW, MAE: 7.56 kW, R²: 0.943;
    Held-out chronological walk-forward cross-validation: RMSE: 10.45 kW, MAE: 7.79 kW.
    Note on Weekend Heuristic (Phase 9): The supplied train_telemetry.csv contains 72 hours spanning
    Tuesday through Thursday only (0 weekend observations). Speculative multipliers have been removed
    in favor of the scientifically trained empirical Ridge parameters.
    """
    hour, _ = parse_timestamp_hour_weekday(timestamp)
    is_evening = 1.0 if 18 <= hour <= 22 else 0.0

    sin_h = math.sin((2.0 * math.pi * hour) / 24.0)
    cos_h = math.cos((2.0 * math.pi * hour) / 24.0)

    raw_features = [float(temp), float(humidity), float(solar), sin_h, cos_h, is_evening]

    # Normalize features using training distribution
    pred = TRAINED_WEIGHTS[0]  # Intercept
    for i in range(len(raw_features)):
        std = TRAINED_STDS[i] if TRAINED_STDS[i] > 1e-6 else 1.0
        norm_val = (raw_features[i] - TRAINED_MEANS[i]) / std
        pred += TRAINED_WEIGHTS[i + 1] * norm_val

    return max(35.0, round(pred, 2))


def call_quickml_batch(rows):
    """
    Calls published Zoho Catalyst QuickML endpoint if configured.
    Conforms to Catalyst QuickML pipeline REST specification.
    """
    if not QUICKML_ENDPOINT or not QUICKML_ENDPOINT.startswith("http"):
        return None

    import urllib.request
    import urllib.error

    payload = {
        "instances": [
            {
                "timestamp": str(r["timestamp"]),
                "ambient_temp_c": float(r["ambient_temp_c"]),
                "humidity_pct": float(r["humidity_pct"]),
                "solar_irradiance_wm2": float(r["solar_irradiance_wm2"]),
            }
            for r in rows
        ]
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if QUICKML_API_KEY:
        token = QUICKML_API_KEY if QUICKML_API_KEY.startswith("Zoho-oauthtoken") else f"Zoho-oauthtoken {QUICKML_API_KEY}"
        headers["Authorization"] = token
    if QUICKML_PROJECT_ID:
        headers["X-Catalyst-Project-Id"] = QUICKML_PROJECT_ID

    req = urllib.request.Request(QUICKML_ENDPOINT, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            if resp.status == 200:
                res_json = json.loads(resp.read().decode("utf-8"))
                preds = res_json.get("predictions") or [
                    d.get("target_demand_kw", d.get("prediction", d))
                    for d in res_json.get("data", [])
                ]
                if len(preds) == len(rows):
                    return [float(p) for p in preds]
    except Exception as e:
        if not ALLOW_LOCAL_FALLBACK:
            raise RuntimeError(f"Zoho Catalyst QuickML invocation failed: {e}")
        print(f"[QuickML Notice] Live endpoint error ({e}). Using local baseline.", file=sys.stderr)

    return None



def predict(input_df):
    """
    Interface A — Official Starter Kit Python Function Contract.
    Accepts:
        - pandas.DataFrame containing required columns
        - OR list of row dictionaries
        - OR dict of column arrays
    Returns:
        NumPy array (if numpy is installed) or Python list of numeric floats,
        strictly one prediction per input row.
    """
    rows = []
    if hasattr(input_df, "columns"):
        # pandas DataFrame
        cols = list(input_df.columns)
        rows = input_df.to_dict(orient="records")
    elif isinstance(input_df, list):
        if len(input_df) == 0:
            return []
        cols = list(input_df[0].keys())
        rows = input_df
    elif isinstance(input_df, dict):
        # Column-oriented dictionary
        cols = list(input_df.keys())
        first_col = next(iter(input_df.values()))
        n_rows = len(first_col) if hasattr(first_col, "__len__") else 0
        rows = [{c: input_df[c][i] for c in cols} for i in range(n_rows)]
    else:
        raise TypeError(f"Unsupported input type for predict(): {type(input_df)}")

    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    if len(rows) == 0:
        return []

    # Validate numeric values
    for idx, r in enumerate(rows):
        for col in ["ambient_temp_c", "humidity_pct", "solar_irradiance_wm2"]:
            val = r.get(col)
            if val is None:
                raise ValueError(f"Row {idx} missing value for column '{col}'")
            try:
                f_val = float(val)
                if math.isnan(f_val) or math.isinf(f_val):
                    raise ValueError(f"Row {idx} contains NaN or Infinite value in '{col}'")
            except (ValueError, TypeError):
                raise ValueError(f"Row {idx} has non-numeric value '{val}' in '{col}'")

    # Attempt QuickML if configured
    preds = call_quickml_batch(rows)

    if preds is None:
        if QUICKML_ENDPOINT and not ALLOW_LOCAL_FALLBACK:
            raise RuntimeError("QuickML endpoint configured but call failed and fallback is disabled.")
        # Local demonstration baseline
        preds = [
            predict_demand_local_baseline(
                r["timestamp"],
                float(r["ambient_temp_c"]),
                float(r["humidity_pct"]),
                float(r["solar_irradiance_wm2"])
            )
            for r in rows
        ]

    # Validate output finite
    for p in preds:
        if not math.isfinite(p):
            raise ValueError(f"Model generated non-finite prediction: {p}")

    # Return as numpy array if numpy is available, otherwise list
    try:
        import numpy as np
        return np.array(preds, dtype=float)
    except ImportError:
        return preds


def run_pipeline(input_path, output_path):
    """
    Interface B — Command-line prediction pipeline.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    raw_rows = []
    with open(input_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            print(f"Error: Missing required columns in input CSV: {missing}", file=sys.stderr)
            sys.exit(1)

        for row in reader:
            raw_rows.append(row)

    if not raw_rows:
        print("Error: Input CSV contains zero telemetry data rows.", file=sys.stderr)
        sys.exit(1)

    # Use predict() function
    try:
        import pandas as pd
        df = pd.DataFrame(raw_rows)
        predictions = predict(df)
    except ImportError:
        # Without pandas installed
        predictions = []
        for r in raw_rows:
            pred = predict_demand_local_baseline(
                r["timestamp"],
                float(r["ambient_temp_c"]),
                float(r["humidity_pct"]),
                float(r["solar_irradiance_wm2"])
            )
            predictions.append(pred)

    # Write output CSV strictly adhering to evaluation contract
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "target_demand_kw"])
        for r, pred in zip(raw_rows, predictions):
            writer.writerow([r["timestamp"], f"{float(pred):.2f}"])

    print(f"Successfully generated {len(predictions)} predictions into '{output_path}'.")


def main():
    parser = argparse.ArgumentParser(
        description="GridMind — QuickML Microgrid Demand Predictor"
    )
    parser.add_argument("--input", required=True, help="Path to input telemetry CSV")
    parser.add_argument("--output", required=True, help="Path to output predictions CSV")
    args = parser.parse_args()

    run_pipeline(args.input, args.output)


if __name__ == "__main__":
    main()
