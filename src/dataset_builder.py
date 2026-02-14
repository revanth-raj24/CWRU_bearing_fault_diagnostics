"""
dataset_builder.py
------------------
Walks data/raw/, loads each .mat file, segments the signal, extracts features,
and saves X.npy / y.npy / groups.npy to data/processed/.

Multi-class labelling:
  Folder names are matched (case-insensitive substring) against LABEL_MAP in
  config.py.  The first key that appears in the folder name wins.  Any folder
  with no match raises a clear ValueError so bad data is caught early.

  Expected layout example:
      data/raw/
          normal/          → label 0
          inner_race/      → label 1
          ball/            → label 2
"""

import os
import numpy as np
from tqdm import tqdm

from src.config import SEGMENT_LENGTH, LABEL_MAP
from src.data_loader import load_mat_file
from src.segmentation import segment_signal
from src.feature_engineering import extract_features


def _folder_to_label(folder_name: str) -> int:
    """
    Map a folder name to an integer class label using LABEL_MAP.

    The folder name is lower-cased and each key in LABEL_MAP is tested as a
    substring.  The first match (in insertion order, Python 3.7+) wins.

    Raises
    ------
    ValueError
        If no key from LABEL_MAP appears in the folder name.
    """
    name_lower = folder_name.lower()
    for key, label in LABEL_MAP.items():
        if key in name_lower:
            return label
    raise ValueError(
        f"Cannot determine class label for folder '{folder_name}'. "
        f"Folder name must contain one of: {list(LABEL_MAP.keys())}. "
        f"Update LABEL_MAP in src/config.py to add new fault types."
    )


def build_dataset(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    segment_length: int = SEGMENT_LENGTH,
) -> None:
    """
    Build and save the processed dataset from raw .mat files.

    Parameters
    ----------
    raw_dir : str
        Root directory containing per-class subdirectories of .mat files.
    processed_dir : str
        Output directory; X.npy, y.npy, groups.npy will be written here.
    segment_length : int
        Number of samples per segment window.

    Output files
    ------------
    X.npy      : float32 array, shape (N_segments, N_features)
    y.npy      : int32 array,   shape (N_segments,)  — class labels
    groups.npy : int32 array,   shape (N_segments,)  — file-level group id
                 Ensures no file appears in both train and test splits.
    """
    os.makedirs(processed_dir, exist_ok=True)

    all_features: list[np.ndarray] = []
    all_labels:   list[int]        = []
    all_groups:   list[int]        = []

    group_id = 0  # incremented per .mat file

    # Collect all (folder, filename) pairs upfront for a clean progress bar
    mat_files: list[tuple[str, str, str]] = []
    for folder_name in sorted(os.listdir(raw_dir)):
        folder_path = os.path.join(raw_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        for fname in sorted(os.listdir(folder_path)):
            if fname.endswith(".mat"):
                mat_files.append((folder_name, folder_path, fname))

    if not mat_files:
        raise RuntimeError(
            f"No .mat files found under '{raw_dir}'. "
            "Check your data/raw/ directory layout."
        )

    for folder_name, folder_path, fname in tqdm(mat_files, desc="Building dataset"):
        label = _folder_to_label(folder_name)
        fpath = os.path.join(folder_path, fname)

        try:
            signal = load_mat_file(fpath)
        except Exception as exc:
            print(f"  [WARN] Skipping {fpath}: {exc}")
            continue

        segments = segment_signal(signal, segment_length)
        if len(segments) == 0:
            print(f"  [WARN] No complete segments from {fpath} (signal length {len(signal)})")
            continue

        for seg in segments:
            feats = extract_features(seg)
            all_features.append(feats)
            all_labels.append(label)
            all_groups.append(group_id)

        group_id += 1

    if not all_features:
        raise RuntimeError("Dataset is empty after processing all .mat files.")

    X      = np.array(all_features, dtype=np.float32)
    y      = np.array(all_labels,   dtype=np.int32)
    groups = np.array(all_groups,   dtype=np.int32)

    np.save(os.path.join(processed_dir, "X.npy"),      X)
    np.save(os.path.join(processed_dir, "y.npy"),      y)
    np.save(os.path.join(processed_dir, "groups.npy"), groups)

    # ── Summary ───────────────────────────────────────────────────────────
    from src.config import CLASS_NAMES
    print(f"\nDataset saved to '{processed_dir}'")
    print(f"  Total segments : {len(y):,}")
    print(f"  Features       : {X.shape[1]}")
    print(f"  Groups (files) : {groups.max() + 1}")
    print("  Class distribution:")
    for cls_id, cls_name in sorted(CLASS_NAMES.items()):
        count = int((y == cls_id).sum())
        print(f"    [{cls_id}] {cls_name:<12} : {count:,} segments")