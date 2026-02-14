# CWRU Bearing Fault Detection — Project Status

**Purpose:** High-level status for handoff to another agent. Describes what exists, what works, and what is missing or inconsistent.

**Last updated:** 2025-02-14

---

## 1. Project overview

- **Domain:** CWRU (Case Western Reserve University) bearing vibration dataset — binary classification: **normal (0)** vs **fault (1)**.
- **Data:** Raw `.mat` files expected under `data/raw/`. Each file should contain a key with `"DE_time"` (drive-end time signal). Folder name `normal` → label 0; any other folder → label 1.
- **Pipeline:** Raw MAT → load → segment → extract features → group-aware split → train Random Forest → evaluate / save model.

---

## 2. Repository layout

```
CWRU/
├── build_data.py          # Entry: build processed dataset from raw .mat
├── train_model.py         # Entry: train + evaluate + save model
├── cross_validation.py    # Entry: 5-fold group cross-validation
├── test_split.py          # Ad-hoc script: check train/test group separation
├── requirements.txt       # Dependencies (incomplete — see below)
├── src/
│   ├── config.py          # SEGMENT_LENGTH, TEST_SIZE, RANDOM_STATE, FEATURE_NAMES
│   ├── data_loader.py     # load_mat_file() — find DE_time key, return 1D signal
│   ├── segmentation.py    # segment_signal() — non-overlapping windows
│   ├── feature_engineering.py  # extract_features() — time + frequency stats
│   ├── dataset_builder.py # build_dataset() — walk raw, segment, featurize, save X,y,groups
│   ├── split.py           # group_train_test_split(), group_cross_validation()
│   ├── train.py           # train_random_forest(), evaluate_model(), save_model()
│   └── evaluate.py        # Empty (placeholder)
└── temporary_debug/
    └── temp.py            # One-off: print .mat keys for 97.mat
```

- No `README`, no formal test suite (e.g. `pytest`), no `data/` committed (paths assume `data/raw/`, `data/processed/`).

---

## 3. Implemented and working

| Component | Status | Notes |
|-----------|--------|--------|
| **Data loading** | Done | `data_loader.py`: auto-detects `DE_time` in `.mat`, returns flattened signal. |
| **Segmentation** | Done | Fixed length (from config), non-overlapping; used in dataset builder. |
| **Feature extraction** | Done | Time: mean, std, rms, skew, kurtosis, crest factor. Freq: fft mean, std, max. |
| **Dataset builder** | Done | Walks `data/raw/`, labels by folder (normal/fault), outputs `X.npy`, `y.npy`, `groups.npy` under `data/processed/`. |
| **Group-aware split** | Done | `GroupShuffleSplit` for train/test; `GroupKFold` for CV so same file/group never in both train and test. |
| **Training** | Done | Random Forest (200 trees, `class_weight="balanced"`), `StandardScaler` fit on train only. |
| **Evaluation (inline)** | Done | In `train.py`: accuracy, classification report, confusion matrix. |
| **Model persistence** | Done | `save_model()` writes `{"model", "scaler"}` via joblib (default `models/rf_baseline.pkl`). |
| **Scripts** | Done | `build_data.py`, `train_model.py`, `cross_validation.py` are wired end-to-end; `test_split.py` checks group disjointness. |

---

## 4. Incomplete or inconsistent

| Item | Details |
|------|--------|
| **`src/evaluate.py`** | Empty. No standalone evaluation script (e.g. load model + test set and print metrics). |
| **Config usage** | Only `SEGMENT_LENGTH` is imported. `TEST_SIZE`, `RANDOM_STATE`, `FEATURE_NAMES` are defined but not used; `split.py` and `train_model.py` use hardcoded `0.2` and `42`. |
| **`requirements.txt`** | Lists only `numpy` and `scipy`. Missing: `scikit-learn`, `tqdm`, `joblib`. |
| **Model load path** | `save_model()` uses default `models/rf_baseline.pkl`; no corresponding “load and run” script or CLI. |
| **Data paths** | All paths are hardcoded (`data/raw/`, `data/processed/`, `models/`). No env or config for paths. |

---

## 5. Assumptions and dependencies

- **Python:** Scripts assume run from project root (e.g. `python build_data.py`) so `from src...` and `data/processed/X.npy` resolve.
- **Data:** Raw CWRU-style `.mat` files under `data/raw/`, with folder-based labels; `DE_time` key must exist.
- **Order of operations:** Run `build_data.py` first to create `data/processed/`; then `train_model.py` or `cross_validation.py` (they load `.npy` from there).
- **Outputs:** Processed data in `data/processed/`; saved model in `models/` (directory must exist or be created when saving).

---

## 6. Suggested next steps (for an agent)

1. **Fix dependencies:** Add `scikit-learn`, `tqdm`, `joblib` to `requirements.txt`.
2. **Use config everywhere:** Have `split.py` and `train_model.py` use `TEST_SIZE` and `RANDOM_STATE` from `config.py`.
3. **Implement `evaluate.py`:** Load saved model + scaler, load test data (or processed data + split), run `evaluate_model()` and optionally save metrics.
4. **Optional:** Add a small README (data source, how to run build → train/cv), and consider path/config via env or config file.
5. **Optional:** Add `pytest` and at least one test (e.g. `group_train_test_split` leaves no group overlap) so `test_split.py` logic is regression-tested.

---

## 7. One-line summary

**Status:** Pipeline from raw CWRU `.mat` to trained Random Forest is implemented and wired via `build_data.py` and `train_model.py`/`cross_validation.py`; config is underused, `evaluate.py` is empty, and `requirements.txt` is incomplete — suitable for local runs and further hardening.
