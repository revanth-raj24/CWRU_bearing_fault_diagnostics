import numpy as np
from scipy.stats import skew, kurtosis

def extract_features(segment):
    features = {}

    # Time domain
    features["mean"] = np.mean(segment)
    features["std"] = np.std(segment)
    features["rms"] = np.sqrt(np.mean(segment**2))
    features["skew"] = skew(segment)
    features["kurtosis"] = kurtosis(segment)
    features["crest_factor"] = np.max(np.abs(segment)) / features["rms"]

    # Frequency domain
    fft_vals = np.abs(np.fft.rfft(segment))
    features["fft_mean"] = np.mean(fft_vals)
    features["fft_std"] = np.std(fft_vals)
    features["fft_max"] = np.max(fft_vals)

    return list(features.values())
