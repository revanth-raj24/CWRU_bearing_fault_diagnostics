
import numpy as np

# Ensure project root is on sys.path so `from src...` works when running
# this test directly from the `tests/` folder.
# sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.split import group_train_test_split

X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")
groups = np.load("data/processed/groups.npy")

X_train, X_test, y_train, y_test, g_train, g_test = \
    group_train_test_split(X, y, groups)

print("Train groups:", np.unique(g_train))
print("Test groups:", np.unique(g_test))

print("Intersection:",
      set(np.unique(g_train)).intersection(set(np.unique(g_test))))
