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


def _graph_cut_refine(img, mask, models, gamma=50.0, iterations=1):
    """
    Simplified Graph Cut refinement.
    Reassigns each pixel to fg or bg based on data term + smoothness term.
    """
    h, w = img.shape[:2]
    result = mask.copy().astype(np.uint8)

    for _ in range(iterations):
        new_mask = result.copy()
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                color = img[y, x].astype(np.float64)
                fg_prob = _pixel_probability(color, models, 'fg')
                bg_prob = _pixel_probability(color, models, 'bg')

                # Smoothness: count neighbors of each label
                neighbors = np.array([
                    result[y-1,x], result[y,x-1], result[y,x+1], result[y+1,x],
                    result[y-1,x-1], result[y-1,x+1], result[y+1,x-1], result[y+1,x+1]
                ])
                n_fg = (neighbors == 1).sum()
                n_bg = (neighbors == 0).sum()

                # Energy: data term + gamma * smoothness term
                energy_fg = -np.log(fg_prob) - gamma * n_fg
                energy_bg = -np.log(bg_prob) - gamma * n_bg

                new_mask[y, x] = 1 if energy_fg < energy_bg else 0
        result = new_mask

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
