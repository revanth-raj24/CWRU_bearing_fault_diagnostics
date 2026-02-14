import numpy as np
from src.train import train_random_forest
from src.split import group_cross_validation

X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")
groups = np.load("data/processed/groups.npy")

group_cross_validation(train_random_forest, X, y, groups, n_splits=5)
