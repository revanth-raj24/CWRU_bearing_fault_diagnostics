"""
evaluate.py
-----------
Standalone evaluation module.

Can be called directly as a script:
    python evaluate.py

Or imported and called programmatically:
    from src.evaluate import run_evaluation
    metrics = run_evaluation()

Loads the saved model + scaler, performs a group-aware train/test split
(same seed as training so the test set is identical), and prints full
multi-class metrics.  Optionally saves metrics to a JSON file.
"""

import json
import os
import numpy as np

from src.config import RANDOM_STATE, TEST_SIZE, CLASS_NAMES
from src.split import group_train_test_split
from src.train import load_model, evaluate_model


def run_evaluation(
    processed_dir: str = "data/processed",
    model_path: str = "models/rf_multiclass.pkl",
    output_json: str | None = "models/eval_metrics.json",
) -> dict:
    """
    Load data + model, split, evaluate, and (optionally) save metrics.

    Parameters
    ----------
    processed_dir : directory containing X.npy, y.npy, groups.npy
    model_path    : path to the saved model bundle (joblib)
    output_json   : if set, write metrics dict to this JSON file

    Returns
    -------
    metrics : dict with "accuracy", "report", "confusion_matrix" (as list)
    """
    # ── Load processed data ───────────────────────────────────────────────
    X      = np.load(os.path.join(processed_dir, "X.npy"))
    y      = np.load(os.path.join(processed_dir, "y.npy"))
    groups = np.load(os.path.join(processed_dir, "groups.npy"))

    # ── Reproduce the same train/test split used during training ──────────
    _, X_test, _, y_test = group_train_test_split(
        X, y, groups,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"Test set  : {len(y_test):,} segments")
    labels      = sorted(CLASS_NAMES.keys())
    target_names = [CLASS_NAMES[l] for l in labels]
    for cls_id, cls_name in zip(labels, target_names):
        count = int((y_test == cls_id).sum())
        print(f"  [{cls_id}] {cls_name:<12}: {count:,}")

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\nLoading model from '{model_path}' …")
    model, scaler = load_model(model_path)

    # ── Evaluate ──────────────────────────────────────────────────────────
    metrics = evaluate_model(model, scaler, X_test, y_test)

    # ── Persist metrics ───────────────────────────────────────────────────
    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        # confusion_matrix is a numpy array — convert to nested list for JSON
        serialisable = {
            "accuracy":         float(metrics["accuracy"]),
            "report":           metrics["report"],
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
        }
        with open(output_json, "w") as f:
            json.dump(serialisable, f, indent=2)
        print(f"\nMetrics saved → {output_json}")

    return metrics


if __name__ == "__main__":
    run_evaluation()