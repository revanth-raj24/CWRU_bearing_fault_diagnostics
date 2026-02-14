import numpy as np
from src.split import group_train_test_split
from src.train import train_random_forest, evaluate_model, save_model

X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")
groups = np.load("data/processed/groups.npy")

X_train, X_test, y_train, y_test, _, _ = \
    group_train_test_split(X, y, groups)

model, scaler = train_random_forest(X_train, y_train)

evaluate_model(model, scaler, X_test, y_test)

save_model(model, scaler)
