"""Simplified GrabCut foreground extraction. Pure NumPy."""
import numpy as np


def _compute_color_model(fg_pixels, bg_pixels):
    """Simple K-means based color model (replaces GMM for demo purposes)."""
    k = 3  # Number of clusters per class
    models = {'fg': [], 'bg': []}

    for name, pixels in [('fg', fg_pixels), ('bg', bg_pixels)]:
        if len(pixels) == 0:
            models[name] = [{'mean': np.array([128,128,128]), 'weight': 1.0}]
            continue
        # Simple K-means
        data = np.array(pixels, dtype=np.float64)
        n = min(k, len(data))
        if n == 0:
            models[name] = [{'mean': np.array([128,128,128]), 'weight': 1.0}]
            continue
        # Initialize with random samples
        indices = np.random.choice(len(data), n, replace=False)
        centroids = data[indices].copy()
        for _ in range(5):
            dists = np.sum((data[:, None] - centroids[None, :])**2, axis=2)
            labels = np.argmin(dists, axis=1)
            for j in range(n):
                if (labels == j).any():
                    centroids[j] = data[labels == j].mean(axis=0)
        models[name] = [{'mean': centroids[j], 'weight': 1.0/n} for j in range(n)]

    return models


def _pixel_probability(color, models, name):
    """Compute probability that a pixel belongs to foreground/background."""
    prob = 0.0
    for comp in models[name]:
        diff = np.array(color, dtype=np.float64) - comp['mean']
        dist = np.sum(diff * diff)
        prob += comp['weight'] * np.exp(-dist / (2.0 * 50.0**2))
    return max(prob, 1e-12)


def _pixel_probability_all(img, models, name):
    """Vectorized: compute probability for all pixels at once."""
    prob = np.zeros(img.shape[:2], dtype=np.float64)
    for comp in models[name]:
        diff = img.astype(np.float64) - comp['mean'].reshape(1, 1, 3)
        dist = np.sum(diff * diff, axis=2)
        prob += comp['weight'] * np.exp(-dist / (2.0 * 50.0**2))
    return np.maximum(prob, 1e-12)


def _graph_cut_refine(img, mask, models, gamma=50.0, iterations=1):
    """Vectorized Graph Cut refinement."""
    h, w = img.shape[:2]
    result = mask.copy().astype(np.uint8)

    for _ in range(iterations):
        # Data term: per-pixel fg/bg probability (vectorized over all pixels)
        fg_prob = _pixel_probability_all(img, models, 'fg')
        bg_prob = _pixel_probability_all(img, models, 'bg')
        data_fg = -np.log(fg_prob)
        data_bg = -np.log(bg_prob)

        # Smoothness: count fg neighbors using convolution
        kernel = np.ones((3, 3), dtype=np.float64)
        kernel[1, 1] = 0  # Don't count self
        from scipy.ndimage import convolve
        n_fg = convolve((result == 1).astype(np.float64), kernel, mode='constant', cval=0.0)
        n_bg = convolve((result == 0).astype(np.float64), kernel, mode='constant', cval=0.0)

        energy_fg = data_fg - gamma * n_fg
        energy_bg = data_bg - gamma * n_bg
        result = np.where(energy_fg < energy_bg, 1, 0).astype(np.uint8)

    return result


def grabcut_segment(img, rect):
    """
    Simplified GrabCut segmentation.

    Args:
        img: RGB image
        rect: (x, y, w, h) bounding box around the object

    Returns: binary mask (255=foreground, 0=background)
    """
    img_u8 = np.asarray(img, dtype=np.uint8)
    if img_u8.ndim == 2:
        img_u8 = np.stack([img_u8]*3, axis=-1)

    x, y, w, h_box = rect
    h_img, w_img = img_u8.shape[:2]
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = max(1, min(w, w_img - x))
    h_box = max(1, min(h_box, h_img - y))

    # Initialize: inside rect = probable fg, outside = definite bg
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    mask[y:y+h_box, x:x+w] = 1  # Probable foreground
    mask[:5, :] = 0
    mask[-5:, :] = 0
    mask[:, :5] = 0
    mask[:, -5:] = 0

    # Collect color samples
    fg_pixels = img_u8[mask == 1].reshape(-1, 3)
    bg_pixels = img_u8[mask == 0].reshape(-1, 3)

    # Build color models
    models = _compute_color_model(fg_pixels, bg_pixels)

    # Refine with simplified graph cut
    refined = _graph_cut_refine(img_u8, mask, models, gamma=30.0, iterations=2)

    return np.where(refined > 0, 255, 0).astype(np.uint8)
