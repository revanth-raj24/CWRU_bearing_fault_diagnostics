"""
cross_validation.py
-------------------
Entry point: stratified group k-fold cross-validation.

Why n_splits=4 (not 5)?
  Normal class has only 4 source .mat files (groups).
  StratifiedGroupKFold needs at least n_splits groups per class to
  guarantee every test fold contains Normal samples.
  Using n_splits=4 is the maximum safe value given the data.

Usage (from project root):
    python cross_validation.py

Prerequisites:
    Run `python build_data.py` first to create data/processed/.
"""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from src.config import CLASS_NAMES, RANDOM_STATE
from src.split import group_cross_validation
from src.train import train_random_forest


def main(n_splits: int = 4):
    # ── Load processed data ───────────────────────────────────────────────
    print("Loading processed data …")
    X      = np.load("data/processed/X.npy")
    y      = np.load("data/processed/y.npy")
    groups = np.load("data/processed/groups.npy")

    labels       = sorted(CLASS_NAMES.keys())
    target_names = [CLASS_NAMES[l] for l in labels]

    # ── Print group distribution upfront so imbalance is visible ─────────
    print(f"\nGroups per class (source .mat files):")
    for cls_id in labels:
        n_groups = len(np.unique(groups[y == cls_id]))
        print(f"  [{cls_id}] {CLASS_NAMES[cls_id]:<10}: {n_groups} groups")
    print(f"\nRunning {n_splits}-fold stratified group CV …")

    folds = group_cross_validation(X, y, groups, n_splits=n_splits)
    fold_accuracies: list[float] = []
    skipped_folds:  list[int]   = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ── Guard: skip fold if any class is missing from test ────────────
        classes_in_test = set(y_test.tolist())
        missing = [CLASS_NAMES[c] for c in labels if c not in classes_in_test]
        if missing:
            print(f"\n{'─'*60}")
            print(f"Fold {fold_idx}/{n_splits}  |  SKIPPED — missing classes in test: {missing}")
            skipped_folds.append(fold_idx)
            continue

        model, scaler = train_random_forest(X_train, y_train, random_state=RANDOM_STATE)
        X_test_scaled = scaler.transform(X_test)
        y_pred        = model.predict(X_test_scaled)

        # ── Per-fold class support summary ────────────────────────────────
        print(f"\n{'─'*60}")
        print(f"Fold {fold_idx}/{n_splits}  |  test support: "
              + ", ".join(f"{CLASS_NAMES[c]}={int((y_test==c).sum())}" for c in labels))

        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)

        print(f"Overall accuracy : {acc:.4f}")
        print(classification_report(
            y_test, y_pred,
            labels=labels,
            target_names=target_names,
            digits=4,
            zero_division=0,
        ))

    # ── Aggregate ─────────────────────────────────────────────────────────
    if not fold_accuracies:
        print("\n[ERROR] All folds were skipped — increase Normal data or reduce n_splits.")
        return

    arr = np.array(fold_accuracies)
    valid = n_splits - len(skipped_folds)
    print(f"\n{'='*60}")
    print(f"Stratified group CV summary  ({valid}/{n_splits} valid folds)")
    print(f"{'='*60}")
    print(f"Mean accuracy : {arr.mean():.4f}")
    print(f"Std  accuracy : {arr.std():.4f}")
    print(f"Per-fold      : {[f'{a:.4f}' for a in fold_accuracies]}")
    if skipped_folds:
        print(f"Skipped folds : {skipped_folds}  (missing class in test — add more Normal files)")


if __name__ == "__main__":
    main()