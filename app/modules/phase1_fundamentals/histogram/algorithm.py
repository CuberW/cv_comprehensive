"""Histogram computation and equalization. Pure NumPy."""
import numpy as np


def compute_histogram(gray_img, bins=256):
    """Count pixels at each brightness level [0, 255]."""
    arr = np.asarray(gray_img, dtype=np.uint8).ravel()
    hist = np.bincount(arr, minlength=bins).astype(np.float64)
    return hist


def compute_rgb_histograms(img):
    """Compute per-channel histograms for R, G, B."""
    arr = np.asarray(img, dtype=np.uint8)
    hists = []
    for c in range(3):
        hists.append(compute_histogram(arr[:, :, c]))
    return hists


def histogram_equalization(gray_img):
    """
    Equalize histogram to spread brightness uniformly.
    Uses the CDF (Cumulative Distribution Function) as the mapping function.
    Formula: new_pixel = round(CDF_normalized(old_pixel) * 255)
    """
    arr = np.asarray(gray_img, dtype=np.uint8)
    hist = compute_histogram(arr)
    cdf = np.cumsum(hist)
    cdf_min = cdf[cdf > 0].min() if (cdf > 0).any() else 0
    cdf_norm = (cdf - cdf_min) / (arr.size - cdf_min)
    mapping = np.round(cdf_norm * 255).astype(np.uint8)
    return mapping[arr]
