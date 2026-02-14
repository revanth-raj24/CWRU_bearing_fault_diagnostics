"""
split.py
--------
Group-aware train/test split and cross-validation utilities.

Using groups ensures that all segments from the same .mat file land
entirely in either train or test — never split across both.  This
prevents data leakage that would inflate performance metrics.

Why StratifiedGroupKFold instead of GroupKFold
-----------------------------------------------
GroupKFold distributes *groups* evenly across folds but ignores class
balance.  When one class has far fewer source files than others (e.g.
Normal has 4 files vs 16 for Inner/Ball), some folds end up with zero
test samples for that class — making per-class metrics meaningless and
overall accuracy artificially inflated.

StratifiedGroupKFold (sklearn >= 1.1) solves this: it respects both
constraints simultaneously — no file leaks across folds AND each fold
gets a representative proportion of every class.
"""

import warnings
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from src.config import TEST_SIZE, RANDOM_STATE


def group_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into train/test with no group (file) overlap.

    Parameters
    ----------
    X, y, groups : arrays of shape (N,) or (N, F)
    test_size    : fraction of groups held out for test (default from config)
    random_state : RNG seed (default from config)

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def group_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Return (train_idx, test_idx) tuples for stratified group k-fold CV.

    Uses StratifiedGroupKFold so that:
      - No .mat file (group) is split across train and test in any fold.
      - Every fold's test set contains a balanced mix of all classes.

    This is critical when class group counts are unequal (e.g. Normal
    has 4 source files while Inner/Ball each have 16).

    Parameters
    ----------
    n_splits : number of folds. Must be <= min groups per class.
               With 4 Normal groups, n_splits <= 4 is safe.

    Returns
    -------
    List of (train_indices, test_indices) tuples, one per fold.
    """
    # Warn early if n_splits exceeds the smallest per-class group count
    unique_classes = np.unique(y)
    min_groups = min(
        len(np.unique(groups[y == c])) for c in unique_classes
    )
    if n_splits > min_groups:
        warnings.warn(
            f"n_splits={n_splits} exceeds the smallest per-class group count "
            f"({min_groups}). Some folds may still have empty test classes. "
            f"Consider setting n_splits <= {min_groups}.",
            UserWarning,
            stacklevel=2,
        )

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return list(sgkf.split(X, y, groups))