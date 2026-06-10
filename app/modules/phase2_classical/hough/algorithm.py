"""Hough Transform for line detection. Pure NumPy."""
import numpy as np


def hough_line_transform(edge_img, theta_res=180, rho_res=None):
    """
    Hough line detection via rho-theta parameter space voting.
    For each edge pixel (x,y), vote for all lines passing through it:
      rho = x*cos(theta) + y*sin(theta)

    Returns: (accumulator, thetas, rhos)
    """
    edges = np.asarray(edge_img, dtype=np.uint8)
    h, w = edges.shape
    diag = int(np.ceil(np.sqrt(h*h + w*w)))
    if rho_res is None:
        rho_res = diag * 2

    thetas = np.linspace(0, np.pi, theta_res, endpoint=False)
    rhos = np.linspace(-diag, diag, rho_res)
    accumulator = np.zeros((rho_res, theta_res), dtype=np.float64)

    # Get edge pixel coordinates
    ys, xs = np.where(edges > 0)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    for x, y in zip(xs, ys):
        rho_vals = x * cos_t + y * sin_t
        rho_idx = np.round((rho_vals + diag) / (2 * diag) * (rho_res - 1)).astype(np.int32)
        rho_idx = np.clip(rho_idx, 0, rho_res - 1)
        for t_idx, r_idx in enumerate(rho_idx):
            accumulator[r_idx, t_idx] += 1

    return accumulator, thetas, rhos


def find_line_peaks(accumulator, thetas, rhos, threshold_ratio=0.5, min_distance=10, max_lines=50):
    """
    Find peaks in Hough accumulator space.
    Each peak = one detected line: (rho, theta, votes)
    """
    acc = np.asarray(accumulator, dtype=np.float64)
    max_val = acc.max() if acc.size else 1.0
    threshold = max_val * threshold_ratio

    # Find local maxima
    peaks = []
    rho_res, theta_res = acc.shape
    for r in range(1, rho_res - 1):
        for t in range(1, theta_res - 1):
            val = acc[r, t]
            if val < threshold:
                continue
            # Check 3x3 neighborhood
            patch = acc[r-1:r+2, t-1:t+2]
            if val == patch.max():
                peaks.append((float(val), int(r), int(t)))

    # Sort by votes descending
    peaks.sort(reverse=True)

    # Filter peaks that are too close
    filtered = []
    for val, r, t in peaks:
        too_close = False
        for _, pr, pt in filtered:
            dr = abs(r - pr)
            dt = min(abs(t - pt), theta_res - abs(t - pt))
            if dr < min_distance and dt < min_distance:
                too_close = True
                break
        if not too_close:
            filtered.append((val, r, t))
        if len(filtered) >= max_lines:
            break

    return [{'rho': float(rhos[r]), 'theta': float(thetas[t]), 'votes': int(val)}
            for val, r, t in filtered]
