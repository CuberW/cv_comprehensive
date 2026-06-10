"""Thresholding algorithms. Pure NumPy: global, adaptive, Otsu."""
import numpy as np


def global_threshold(gray_img, threshold=128):
    """Simple global threshold: pixel >= t -> 255, else 0."""
    arr = np.asarray(gray_img, dtype=np.uint8)
    return np.where(arr >= threshold, 255, 0).astype(np.uint8)


def adaptive_mean_threshold(gray_img, block_size=11, C=2):
    """
    Adaptive threshold using local mean.
    For each pixel: threshold = mean(neighborhood) - C.
    Works well for images with uneven lighting.
    """
    arr = np.asarray(gray_img, dtype=np.float64)
    h, w = arr.shape
    pad = block_size // 2
    padded = np.pad(arr, pad, mode='edge')
    result = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            patch = padded[i:i+block_size, j:j+block_size]
            thresh = patch.mean() - C
            result[i, j] = 255 if arr[i, j] > thresh else 0
    return result


def otsu_threshold(gray_img):
    """
    Otsu's method: find threshold that maximizes between-class variance.
    Max between-class variance = w0*w1*(m0-m1)^2
    where w0, w1 = class probabilities, m0, m1 = class means.
    """
    arr = np.asarray(gray_img, dtype=np.uint8).ravel()
    hist = np.bincount(arr, minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128

    w0 = 0.0
    sum0 = 0.0
    sum_total = np.dot(np.arange(256), hist)
    max_between = 0.0
    best_t = 0

    for t in range(256):
        w0 += hist[t]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        sum0 += t * hist[t]
        m0 = sum0 / w0
        m1 = (sum_total - sum0) / w1
        between = w0 * w1 * (m0 - m1) ** 2
        if between > max_between:
            max_between = between
            best_t = t

    return best_t
