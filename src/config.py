"""
Central configuration for the CWRU bearing fault detection pipeline.
All scripts should import constants from here — never hardcode these values.
"""

# ── Signal segmentation ────────────────────────────────────────────────────
SEGMENT_LENGTH: int = 1024   # samples per window

# ── Train / test split ────────────────────────────────────────────────────
TEST_SIZE: float = 0.2       # fraction held out for evaluation
RANDOM_STATE: int = 42       # reproducibility seed

# ── Multi-class label map ─────────────────────────────────────────────────
# Maps folder-name substrings (lower-cased) → integer class label.
# Rules are checked in order; first match wins.
# Folder naming convention expected under data/raw/:
#   normal/          → class 0
#   inner_race/  (or any folder containing "inner") → class 1
#   ball/        (or any folder containing "ball")  → class 2
#
# Add or reorder entries here to support additional fault types without
# touching any other source file.
LABEL_MAP: dict[str, int] = {
    "normal": 0,
    "inner":  1,   # matches "inner_race", "innerrace", "IR", etc.
    "ball":   2,
}

CLASS_NAMES: dict[int, str] = {v: k.capitalize() for k, v in LABEL_MAP.items()}
# → {0: "Normal", 1: "Inner", 2: "Ball"}

NUM_CLASSES: int = len(LABEL_MAP)  # 3

# ── Feature names (must match extract_features output order) ──────────────
FEATURE_NAMES: list[str] = [
    "mean",
    "std",
    "rms",
    "skewness",
    "kurtosis",
    "crest_factor",
    "fft_mean",
    "fft_std",
    "fft_max",
]