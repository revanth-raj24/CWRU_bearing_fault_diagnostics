## CWRU Bearing Fault Classification (Multi‑Class)

This repository implements a full pipeline for **multi‑class bearing fault diagnosis** on the Case Western Reserve University (CWRU) vibration dataset.  
Raw `.mat` vibration signals are segmented, transformed into engineered features, and used to train and evaluate a multi‑class Random Forest classifier with strict group‑aware splits (no file leakage between train and test).

### Project structure

- **Top‑level scripts**
  - `build_data.py`: Build the processed dataset (`X.npy`, `y.npy`, `groups.npy`) from `data/raw/`.
  - `train_model.py`: Train a Random Forest classifier on the processed data and save it to `models/rf_multiclass.pkl`.
  - `cross_validation.py`: Run stratified group \(k\)-fold cross‑validation and print per‑fold metrics.
  - `diagnose.py`: Run diagnostic checks for data leakage, group imbalance, and feature separability.
  - `explore_data.ipynb`: Optional notebook for interactive data inspection and plotting.
- **Core library code (`src/`)**
  - `config.py`: Central configuration (segment length, label map, random seeds, feature names).
  - `dataset_builder.py`: Walk `data/raw/`, load `.mat` files, segment signals, extract features, and save processed arrays.
  - `data_loader.py`: Load CWRU `.mat` files and extract the `*DE_time` vibration channel.
  - `segmentation.py`: Fixed‑length windowing of 1‑D vibration signals.
  - `feature_engineering.py`: Time‑ and frequency‑domain feature extraction (mean, std, RMS, skew, kurtosis, crest factor, FFT stats).
  - `train.py`: Model training, evaluation (accuracy, classification report, confusion matrix), and model persistence utilities.
  - `split.py`: Group‑aware train/test split and stratified group cross‑validation helpers.

### Requirements and setup

- **Python**: 3.10+ recommended.
- **Install dependencies** (from the project root):

```bash
pip install -r requirements.txt
```

### Data layout

Place the CWRU `.mat` files under `data/raw/` in per‑class subfolders. Folder names are mapped to integer labels via `LABEL_MAP` in `src/config.py` using **case‑insensitive substring matching** (first match wins).  
Example layout (default 4‑class map in `config.py`):

```text
data/
  raw/
    normal/        # → label 0 ("Normal")
    inner_race/    # → label 1 ("Inner")
    ball/          # → label 2 ("Ball")
    outer_race/    # → label 3 ("Outer")
```

If a folder name does not contain any of the keys in `LABEL_MAP`, `dataset_builder.py` will raise a clear error so mis‑named folders are caught early.

### End‑to‑end workflow

Run all commands from the project root (`CWRU` directory).

- **1. Build processed dataset**

```bash
python build_data.py
```

This walks `data/raw/`, segments each signal into windows of length `SEGMENT_LENGTH` (see `src/config.py`), computes features, and writes:

- `data/processed/X.npy` – shape \((N_{\text{segments}}, N_{\text{features}})\)
- `data/processed/y.npy` – integer labels
- `data/processed/groups.npy` – file‑level group IDs (used to prevent leakage)

- **2. Train final model**

```bash
python train_model.py
```

This performs a **group‑aware train/test split**, trains a Random Forest with class‑balanced weights, prints a full multi‑class report, and saves:

- `models/rf_multiclass.pkl` – `{"model": RandomForestClassifier, "scaler": StandardScaler}`

- **3. Cross‑validation (recommended for reporting)**

```bash
python cross_validation.py
```

This runs stratified group \(k\)-fold CV (default `n_splits=3`, constrained by the number of available groups per class), skips any fold whose test set is missing a class, and prints per‑fold metrics plus summary statistics (mean/std accuracy).

- **4. Diagnostics (leakage & sanity checks)**

```bash
python diagnose.py
```

This script:

- Summarises groups and segment counts per class.
- Verifies there is **no group overlap** between train and test splits.
- Checks for near‑duplicate feature vectors across splits.
- Prints per‑class feature means to visualise separability.

### Configuration

Most behaviour is controlled through `src/config.py`:

- **Signal/feature settings**: `SEGMENT_LENGTH`, `FEATURE_NAMES`.
- **Splitting / randomness**: `TEST_SIZE`, `RANDOM_STATE`.
- **Label map**: `LABEL_MAP` and derived `CLASS_NAMES` / `NUM_CLASSES`.

To add or rename fault types, edit `LABEL_MAP` only; the rest of the pipeline will pick up the new classes automatically (assuming your `data/raw/` subfolder names match the configured keys).

### Reproducibility and notes

- Random seeds (`RANDOM_STATE`) are set centrally in `config.py` and used consistently across splitting and training.
- Group IDs in `groups.npy` ensure no `.mat` file appears in both train and test sets.
- For exploratory analysis and plots, open `explore_data.ipynb` after running `build_data.py`.
