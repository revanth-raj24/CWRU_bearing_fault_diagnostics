"""
Diagnostic script — paste and run from CWRU project root.
Checks 4 common causes of suspiciously perfect accuracy.
"""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from src.config import TEST_SIZE, RANDOM_STATE, CLASS_NAMES

X      = np.load("data/processed/X.npy")
y      = np.load("data/processed/y.npy")
groups = np.load("data/processed/groups.npy")

print("="*60)
print("DIAGNOSTIC REPORT")
print("="*60)

# ── 1. Group distribution per class ──────────────────────────────
print("\n[1] Groups per class:")
for cls_id, cls_name in CLASS_NAMES.items():
    mask = y == cls_id
    g = groups[mask]
    print(f"  [{cls_id}] {cls_name:<10}: {mask.sum():>5} segments, "
          f"{len(set(g.tolist())):>3} unique groups  "
          f"(groups: {sorted(set(g.tolist()))})")

# ── 2. Check train/test group overlap ────────────────────────────
print("\n[2] Train/test group overlap check:")
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups))
train_groups = set(groups[train_idx].tolist())
test_groups  = set(groups[test_idx].tolist())
overlap = train_groups & test_groups
print(f"  Train groups : {sorted(train_groups)}")
print(f"  Test  groups : {sorted(test_groups)}")
print(f"  Overlap      : {overlap}  ({'LEAK!' if overlap else 'OK'})")

# ── 3. Check for near-duplicate feature rows ─────────────────────
print("\n[3] Near-duplicate feature check:")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
# Sample 500 random pairs and check if any train segment is nearly
# identical to a test segment
rng = np.random.default_rng(42)
train_sample = Xs[train_idx][rng.integers(0, len(train_idx), 2000)]
test_sample  = Xs[test_idx][rng.integers(0, len(test_idx), 500)]
dists = np.linalg.norm(train_sample[:, None] - test_sample[None, :], axis=-1)
min_dist = dists.min()
near_dups = (dists < 0.01).sum()
print(f"  Min L2 distance (train vs test, scaled): {min_dist:.6f}")
print(f"  Pairs with distance < 0.01             : {near_dups}")
if near_dups > 0:
    print("  ⚠ Near-identical feature vectors exist across train/test!")
else:
    print("  OK — no near-duplicates found in sample.")

# ── 4. Feature variance check ────────────────────────────────────
print("\n[4] Feature separability (mean per class):")
labels     = sorted(CLASS_NAMES.keys())
feat_names = ["mean","std","rms","skew","kurt","crest","fft_mean","fft_std","fft_max"]
print(f"  {'Feature':<12}" + "".join(f"{CLASS_NAMES[l]:>12}" for l in labels))
for i, fname in enumerate(feat_names):
    row = f"  {fname:<12}"
    for l in labels:
        row += f"{X[y==l, i].mean():>12.4f}"
    print(row)

print("\n[5] Segments per group (first 10):")
for g in sorted(set(groups.tolist()))[:10]:
    cnt = (groups == g).sum()
    cls = y[groups == g][0]
    print(f"  Group {g:>3}: {cnt:>4} segments  class={CLASS_NAMES[cls]}")