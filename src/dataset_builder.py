import os
import numpy as np
from tqdm import tqdm

from src.data_loader import load_mat_file
from src.segmentation import segment_signal
from src.feature_engineering import extract_features
from src.config import SEGMENT_LENGTH


def build_dataset(raw_data_path, save_path):
    X = []
    y = []
    groups = []

    for root, dirs, files in os.walk(raw_data_path):
        for file in files:
            if not file.endswith(".mat"):
                continue

            filepath = os.path.join(root, file)

            # Label from folder name
            folder_name = os.path.basename(root)

            if folder_name == "normal":
                label = 0
            else:
                label = 1  # fault

            signal = load_mat_file(filepath)
            segments = segment_signal(signal, SEGMENT_LENGTH)

            group_id = file.split(".")[0]

            for segment in segments:
                features = extract_features(segment)

                X.append(features)
                y.append(label)
                groups.append(group_id)

    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)

    os.makedirs(save_path, exist_ok=True)

    np.save(os.path.join(save_path, "X.npy"), X)
    np.save(os.path.join(save_path, "y.npy"), y)
    np.save(os.path.join(save_path, "groups.npy"), groups)

    print("Dataset built successfully.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Class distribution:", np.bincount(y))
    print("Unique groups:", np.unique(groups))
