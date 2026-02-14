import os
import scipy.io
import numpy as np

def load_mat_file(filepath):
    mat = scipy.io.loadmat(filepath)

    # Automatically find the vibration signal key
    for key in mat.keys():
        if "DE_time" in key:
            return mat[key].flatten()

    raise ValueError(f"No DE_time signal found in {filepath}")
