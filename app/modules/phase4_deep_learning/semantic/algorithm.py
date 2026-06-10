"""Simplified semantic segmentation (U-Net style demo). Pure NumPy."""
import numpy as np


def simple_segmentation_demo(img, num_classes=5):
    """
    Demo semantic segmentation using color-based clustering
    as a simplified approximation of what a U-Net would learn.

    Real U-Net uses encoder-decoder with skip connections.
    This demo uses K-means in color space + spatial smoothing
    to produce a visually meaningful segmentation.

    Returns: segmentation map (H x W integer labels)
    """
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    arr = arr / 255.0
    h, w = arr.shape[:2]

    # Downsample for efficiency
    scale = min(1.0, 128.0 / max(h, w))
    if scale < 1.0:
        nh = max(1, int(h * scale))
        nw = max(1, int(w * scale))
        ys = np.linspace(0, h-1, nh).astype(np.int32)
        xs = np.linspace(0, w-1, nw).astype(np.int32)
        small = arr[ys[:, None], xs[None, :]]
    else:
        small = arr

    sh, sw = small.shape[:2]
    pixels = small.reshape(-1, 3)

    # Simple K-means clustering in RGB space
    n_pixels = len(pixels)
    step = max(1, n_pixels // 200)
    sampled = pixels[::step]

    # Initialize centroids
    rng = np.random.default_rng(2026)
    indices = rng.choice(len(sampled), num_classes, replace=False)
    centroids = sampled[indices].copy()

    # K-means iterations
    for _ in range(10):
        dists = np.sum((pixels[:, None] - centroids[None, :])**2, axis=2)
        labels = np.argmin(dists, axis=1)
        for j in range(num_classes):
            if (labels == j).any():
                centroids[j] = pixels[labels == j].mean(axis=0)

    # Assign labels
    dists = np.sum((pixels[:, None] - centroids[None, :])**2, axis=2)
    small_labels = np.argmin(dists, axis=1).reshape(sh, sw)

    # Upsample back to original size
    if scale < 1.0:
        ys_full = np.linspace(0, sh-1, h).astype(np.int32)
        xs_full = np.linspace(0, sw-1, w).astype(np.int32)
        seg_map = small_labels[ys_full[:, None], xs_full[None, :]]
    else:
        seg_map = small_labels

    return seg_map.astype(np.int32)


def colorize_segmentation(seg_map, num_classes=5):
    """Colorize a segmentation map for visualization."""
    colors = np.array([
        [220, 38, 38],   # Red
        [37, 99, 235],   # Blue
        [5, 150, 105],   # Green
        [217, 119, 6],   # Orange
        [124, 58, 237],  # Purple
    ], dtype=np.uint8)

    h, w = seg_map.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(min(num_classes, len(colors))):
        result[seg_map == c] = colors[c]
    return result
