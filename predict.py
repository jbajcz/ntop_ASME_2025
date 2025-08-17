#!/usr/bin/env python3
"""
predict.py

Small CLI utility to load the trained model and scalers (pickles) and predict outputs
for user-provided input parameters: X Cell Size, YZ Cell Size, and Inlet Velocity.

Usage examples (PowerShell):
  python .\predict.py --x 12.5 --yz 15 --v 3000
  python .\predict.py --x 12.5 --yz 15 --v 3000 --json

The script validates inputs against the bounds used in the notebook and prints
the predicted outputs (Pressure Drop, Avg Velocity, Surface Area, Mass).

Contract:
- Inputs: three floats (X cell size mm, YZ cell size mm, inlet velocity mm/s)
- Outputs: JSON or plain text mapping target names to predicted values (unscaled units)
- Errors: useful messages when artifacts are missing or inputs invalid

"""
import argparse
import json
import os
import pickle
import sys
from typing import Tuple

import numpy as np
import pandas as pd

# Targets and default feature order used in the notebooks
TARGETS = ["PressureDrop", "AvgVelocity", "Surface Area", "Mass"]
FEATURES = ["X Cell Size", "YZ Cell Size", "Velocity Inlet"]

# Bounds from the notebook
BOUNDS = {
    "X Cell Size": (10.0, 25.0),
    "YZ Cell Size": (10.0, 25.0),
    "Velocity Inlet": (2500.0, 3500.0),
}


def load_artifacts(model_path: str = "trained_model.pkl", scaler_x_path: str = "scaler_X.pkl", scaler_y_path: str = "scaler_y.pkl"):
    """Load model and scalers from pickle files.

    Returns (model, scaler_X, scaler_y)
    """
    for p in (model_path, scaler_x_path, scaler_y_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required artifact not found: {p}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_x_path, "rb") as f:
        scaler_X = pickle.load(f)

    with open(scaler_y_path, "rb") as f:
        scaler_y = pickle.load(f)

    return model, scaler_X, scaler_y


def predict_outputs(x: np.ndarray, model, scaler_X, scaler_y) -> dict:
    """Predict outputs for a single input vector x (shape: (3,) or (1,3)).

    Returns a dict mapping TARGETS to floats.
    """
    x = np.asarray(x).reshape(1, -1)

    # Try to construct DataFrame using scaler's feature names if available
    try:
        cols = list(scaler_X.feature_names_in_)
        df = pd.DataFrame(x, columns=cols)
    except Exception:
        # Fallback to the expected FEATURES order
        df = pd.DataFrame(x, columns=FEATURES)

    x_scaled = scaler_X.transform(df)

    # model may be a Keras model or a callable; try to use predict
    try:
        y_scaled = model.predict(x_scaled, verbose=0)
    except TypeError:
        # Some pickled callables may require a plain call
        y_scaled = model(x_scaled)

    y = scaler_y.inverse_transform(y_scaled)
    y = np.asarray(y).reshape(-1)

    return {name: float(val) for name, val in zip(TARGETS, y)}


def validate_inputs(x_val: float, yz_val: float, v_val: float) -> Tuple[bool, str]:
    """Validate input ranges. Returns (is_valid, message)."""
    if not (BOUNDS["X Cell Size"][0] <= x_val <= BOUNDS["X Cell Size"][1]):
        return False, f"X Cell Size must be in [{BOUNDS['X Cell Size'][0]}, {BOUNDS['X Cell Size'][1]}]"
    if not (BOUNDS["YZ Cell Size"][0] <= yz_val <= BOUNDS["YZ Cell Size"][1]):
        return False, f"YZ Cell Size must be in [{BOUNDS['YZ Cell Size'][0]}, {BOUNDS['YZ Cell Size'][1]}]"
    if not (BOUNDS["Velocity Inlet"][0] <= v_val <= BOUNDS["Velocity Inlet"][1]):
        return False, f"Velocity Inlet must be in [{BOUNDS['Velocity Inlet'][0]}, {BOUNDS['Velocity Inlet'][1]}]"
    return True, ""


def parse_args():
    p = argparse.ArgumentParser(description="Load trained model and predict outputs for given inputs.")
    p.add_argument("--x", type=float, help="X Cell Size (mm)")
    p.add_argument("--yz", type=float, help="YZ Cell Size (mm)")
    p.add_argument("--v", "--velocity", dest="v", type=float, help="Inlet Velocity (mm/s)")
    p.add_argument("--json", action="store_true", help="Output results as JSON")
    p.add_argument("--no-validate", action="store_true", help="Skip bounds validation (use with caution)")
    p.add_argument("--model", default="trained_model.pkl", help="Path to trained model pickle")
    p.add_argument("--scaler_x", default="scaler_X.pkl", help="Path to scaler_X pickle")
    p.add_argument("--scaler_y", default="scaler_y.pkl", help="Path to scaler_y pickle")
    return p.parse_args()


def main():
    args = parse_args()

    # Collect inputs either from args or interactively
    if args.x is None or args.yz is None or args.v is None:
        print("Some or all inputs missing; entering interactive prompt (press Enter to cancel).")
        try:
            if args.x is None:
                args.x = float(input(f"Enter X Cell Size (mm) [{BOUNDS['X Cell Size'][0]}-{BOUNDS['X Cell Size'][1]}]: "))
            if args.yz is None:
                args.yz = float(input(f"Enter YZ Cell Size (mm) [{BOUNDS['YZ Cell Size'][0]}-{BOUNDS['YZ Cell Size'][1]}]: "))
            if args.v is None:
                args.v = float(input(f"Enter Inlet Velocity (mm/s) [{BOUNDS['Velocity Inlet'][0]}-{BOUNDS['Velocity Inlet'][1]}]: "))
        except (KeyboardInterrupt, EOFError, ValueError):
            print("Input cancelled or invalid. Exiting.")
            sys.exit(1)

    # Validate inputs unless disabled
    if not args.no_validate:
        valid, msg = validate_inputs(args.x, args.yz, args.v)
        if not valid:
            print(f"Input validation failed: {msg}")
            print("Use --no-validate to bypass this check.")
            sys.exit(2)

    # Load artifacts
    try:
        model, scaler_X, scaler_y = load_artifacts(args.model, args.scaler_x, args.scaler_y)
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(3)
    except Exception as e:
        print(f"Failed to load artifacts: {e}")
        sys.exit(4)

    # Build input and predict
    input_vec = np.array([args.x, args.yz, args.v], dtype=float)
    try:
        outputs = predict_outputs(input_vec, model, scaler_X, scaler_y)
    except Exception as e:
        print(f"Prediction failed: {e}")
        sys.exit(5)

    # Print results
    if args.json:
        print(json.dumps({"inputs": {"X Cell Size": args.x, "YZ Cell Size": args.yz, "Velocity Inlet": args.v}, "predictions": outputs}, indent=2))
    else:
        print("\nPredicted outputs (unscaled units):")
        for k, v in outputs.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
