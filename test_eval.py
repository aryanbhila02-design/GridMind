#!/usr/bin/env python3
"""
Official HackSphere 3.0 Evaluation Test Runner
Verifies both submission contracts:
1. Interface A: `from predict import predict; preds = predict(df)`
2. Interface B: CLI `python predict.py --input <in.csv> --output <out.csv>`
3. Schema & integrity verification:
   - Header is exactly: ['timestamp', 'target_demand_kw']
   - Row count matches input row count
   - Timestamps match input timestamps in exact order
   - Predictions are valid numeric floats within realistic physical limits
"""

import csv
import os
import sys

def test_interface_a_function():
    print("[TEST 1/3] Verifying Interface A: `from predict import predict` function contract...")
    try:
        from predict import predict
    except ImportError as e:
        print(f"FAILED: Could not import `predict` from `predict.py`: {e}")
        return False

    # Test with pandas or dictionary input
    sample_records = [
        {"timestamp": "2026-10-27T00:00:00", "ambient_temp_c": 19.2, "humidity_pct": 74, "solar_irradiance_wm2": 0},
        {"timestamp": "2026-10-27T12:00:00", "ambient_temp_c": 30.8, "humidity_pct": 40, "solar_irradiance_wm2": 940},
        {"timestamp": "2026-10-27T19:00:00", "ambient_temp_c": 23.1, "humidity_pct": 68, "solar_irradiance_wm2": 0},
    ]

    try:
        import pandas as pd
        df = pd.DataFrame(sample_records)
        preds = predict(df)
    except ImportError:
        preds = predict(sample_records)

    # Check length
    if len(preds) != len(sample_records):
        print(f"FAILED: Expected {len(sample_records)} predictions, got {len(preds)}")
        return False

    # Check finite floats
    for i, p in enumerate(preds):
        try:
            val = float(p)
            if val < 0 or val > 2000:
                print(f"FAILED: Prediction {i} out of bounds: {val}")
                return False
        except (ValueError, TypeError):
            print(f"FAILED: Prediction {i} is not a valid float: {p}")
            return False

    # Test single-row input
    single_rec = [{"timestamp": "2026-10-27T14:00:00", "ambient_temp_c": 28.0, "humidity_pct": 50, "solar_irradiance_wm2": 600}]
    try:
        import pandas as pd
        p_single = predict(pd.DataFrame(single_rec))
    except ImportError:
        p_single = predict(single_rec)

    if len(p_single) != 1:
        print(f"FAILED: Single row test failed. Expected 1 prediction, got {len(p_single)}")
        return False

    print("SUCCESS: Interface A (function contract) verified!")
    return True


def test_interface_b_cli(input_file="sample_test_input.csv", output_file="sample_predictions.csv"):
    print(f"[TEST 2/3] Verifying Interface B: CLI CSV output '{output_file}' against '{input_file}'...")

    if not os.path.exists(output_file):
        print(f"FAILED: Output file '{output_file}' does not exist.")
        return False

    with open(input_file, mode="r", encoding="utf-8") as f:
        in_reader = csv.DictReader(f)
        in_rows = list(in_reader)

    with open(output_file, mode="r", encoding="utf-8") as f:
        out_reader = csv.reader(f)
        out_rows = list(out_reader)

    if len(out_rows) < 2:
        print("FAILED: Output file has less than 2 rows (header + data).")
        return False

    header = out_rows[0]
    expected_header = ["timestamp", "target_demand_kw"]
    if header != expected_header:
        print(f"FAILED: Invalid header. Expected {expected_header}, got {header}")
        return False

    data_rows = out_rows[1:]
    if len(data_rows) != len(in_rows):
        print(f"FAILED: Row count mismatch. Input has {len(in_rows)}, output has {len(data_rows)}.")
        return False

    for idx, (in_r, out_r) in enumerate(zip(in_rows, data_rows)):
        in_ts = in_r["timestamp"]
        out_ts = out_r[0]
        if in_ts != out_ts:
            print(f"FAILED: Row {idx+1} timestamp mismatch! Input '{in_ts}', output '{out_ts}'.")
            return False

        try:
            val = float(out_r[1])
            if val < 0 or val > 2000:
                print(f"FAILED: Row {idx+1} predicted demand out of bounds: {val} kW")
                return False
        except ValueError:
            print(f"FAILED: Row {idx+1} target_demand_kw is non-numeric: '{out_r[1]}'")
            return False

    print(f"SUCCESS: Interface B passed! All {len(data_rows)} predictions verified.")
    return True


def test_error_handling():
    print("[TEST 3/3] Verifying error handling for missing columns...")
    from predict import predict
    bad_record = [{"timestamp": "2026-10-27T00:00:00", "ambient_temp_c": 19.2}]
    try:
        try:
            import pandas as pd
            test_inp = pd.DataFrame(bad_record)
        except ImportError:
            test_inp = bad_record
        predict(test_inp)
        print("FAILED: Did not raise ValueError on missing weather columns.")
        return False
    except ValueError:
        print("SUCCESS: Error handling correctly rejected missing columns.")
        return True
    except Exception as e:
        print(f"FAILED: Raised unexpected exception: {e}")
        return False


def main():
    inp = sys.argv[1] if len(sys.argv) > 1 else "sample_test_input.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "sample_predictions.csv"

    # Ensure output file is generated via CLI first
    ret = os.system(f"python3 predict.py --input {inp} --output {out}")
    if ret != 0:
        print(f"FAILED: `python3 predict.py --input {inp} --output {out}` exited with non-zero code {ret}.")
        sys.exit(1)

    t1 = test_interface_a_function()
    t2 = test_interface_b_cli(inp, out)
    t3 = test_error_handling()

    if t1 and t2 and t3:
        print("\nALL HACKSPHERE 3.0 SUBMISSION CONTRACT CHECKS PASSED!")
        sys.exit(0)
    else:
        print("\nONE OR MORE CHECKS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
