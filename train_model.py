"""
train_model.py
--------------
Three-class bearing fault classification: Normal (0), Inner race (1), Ball (2).
Loads processed data, group-aware train/test split, trains Random Forest, evaluates, saves model.
"""
import numpy as np
from src.config import CLASS_NAMES
from src.split import group_train_test_split
from src.train import train_random_forest, evaluate_model, save_model

X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")
groups = np.load("data/processed/groups.npy")

# Classes: 0=Normal, 1=Inner, 2=Ball (from data/raw folder layout + config.LABEL_MAP)
print(f"Classes: {CLASS_NAMES}")

X_train, X_test, y_train, y_test = group_train_test_split(X, y, groups)

model, scaler = train_random_forest(X_train, y_train)

evaluate_model(model, scaler, X_test, y_test)

save_model(model, scaler)
