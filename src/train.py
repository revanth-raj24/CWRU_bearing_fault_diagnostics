"""
train.py
--------
Train a Random Forest classifier, evaluate it with multi-class metrics,
and persist the fitted model + scaler to disk.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.config import RANDOM_STATE, CLASS_NAMES


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    random_state: int = RANDOM_STATE,
) -> tuple[RandomForestClassifier, StandardScaler]:
    """
    Fit a StandardScaler on X_train, then train a Random Forest.

    class_weight="balanced" adjusts for any class imbalance automatically,
    which is important now that we have three classes with potentially
    different segment counts.

    Returns
    -------
    model   : fitted RandomForestClassifier
    scaler  : fitted StandardScaler (must be used to transform test data)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,          # use all CPU cores
    )
    model.fit(X_scaled, y_train)
    return model, scaler


def evaluate_model(
    model: RandomForestClassifier,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate the model on test data and print a full multi-class report.

    Returns
    -------
    metrics : dict with keys "accuracy", "report", "confusion_matrix"
    """
    X_scaled = scaler.transform(X_test)
    y_pred   = model.predict(X_scaled)

    acc      = accuracy_score(y_test, y_pred)
    # Build ordered list of class names matching label integers 0, 1, 2, …
    labels      = sorted(CLASS_NAMES.keys())
    target_names = [CLASS_NAMES[l] for l in labels]

    report = classification_report(
        y_test, y_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
    )
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print(f"\nOverall accuracy : {acc:.4f}")
    print("\nPer-class report:")
    print(report)
    print("Confusion matrix (rows=true, cols=predicted):")
    _print_confusion_matrix(cm, target_names)

    return {"accuracy": acc, "report": report, "confusion_matrix": cm}


def _print_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> None:
    """Pretty-print a confusion matrix with row/column headers."""
    col_w = max(len(n) for n in class_names) + 2
    header = " " * col_w + "".join(f"{n:>{col_w}}" for n in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = f"{name:<{col_w}}" + "".join(f"{cm[i, j]:>{col_w}}" for j in range(len(class_names)))
        print(row)


def save_model(
    model: RandomForestClassifier,
    scaler: StandardScaler,
    path: str = "models/rf_multiclass.pkl",
) -> None:
    """
    Persist model and scaler together so they can be loaded as a unit.

    Also saves a human-readable label map alongside the pickle.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, path)
    print(f"\nModel saved → {path}")


def load_model(path: str = "models/rf_multiclass.pkl") -> tuple[RandomForestClassifier, StandardScaler]:
    """
    Load a previously saved model + scaler.

    Returns
    -------
    model, scaler
    """
    bundle = joblib.load(path)
    return bundle["model"], bundle["scaler"]