from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import accuracy_score
import numpy as np


def group_train_test_split(X, y, groups, test_size=0.2, random_state=42):
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    for train_idx, test_idx in gss.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]

    return X_train, X_test, y_train, y_test, groups_train, groups_test

def group_cross_validation(model_fn, X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)

    accuracies = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model, scaler = model_fn(X_train, y_train)

        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        print(f"Fold {fold+1} Accuracy: {acc:.4f}")

    print("\nMean CV Accuracy:", np.mean(accuracies))
    print("Std CV Accuracy:", np.std(accuracies))

    return accuracies
