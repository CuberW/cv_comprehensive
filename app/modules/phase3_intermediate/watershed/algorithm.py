"""Watershed segmentation algorithm. Pure NumPy."""
import numpy as np
from collections import deque


def watershed_segmentation(gradient_img, markers):
    """
    Watershed segmentation by flooding.
    gradient_img: edge strength map (low=basin, high=ridge)
    markers: labeled seed regions (0=unknown, >0=seed labels)

    Floods from each marker, filling basins until they meet at watershed lines.
    Returns: labeled segmentation map
    """
    grad = np.asarray(gradient_img, dtype=np.float64)
    labels = np.asarray(markers, dtype=np.int32).copy()
    h, w = grad.shape

    # Priority queue sorted by gradient value (lowest first = flood from bottom)
    pq = deque()
    for y in range(h):
        for x in range(w):
            if labels[y, x] > 0:
                pq.append((grad[y, x], y, x, labels[y, x]))

    pq = deque(sorted(pq, key=lambda v: v[0]))

    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    visited = (labels > 0)

    while pq:
        _, y, x, lbl = pq.popleft()
        for dy, dx in dirs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                visited[ny, nx] = True
                labels[ny, nx] = lbl
                pq.append((grad[ny, nx], ny, nx, lbl))

    return labels


def make_markers_using_distance(binary_img, min_dist=10):
    """
    Create markers for watershed using distance transform.
    Foreground markers = local maxima of distance transform.
    Background markers = pixels far from any foreground.
    """
    from scipy.ndimage import distance_transform_edt
    binary = np.asarray(binary_img, dtype=np.uint8)
    binary = np.where(binary > 127, 1, 0).astype(np.uint8)

    # Distance transform: each foreground pixel -> distance to nearest background
    dist = distance_transform_edt(binary)

    # Foreground markers: local maxima of distance
    from scipy.ndimage import maximum_filter
    local_max = (dist == maximum_filter(dist, size=min_dist*2+1))
    fg_markers = np.where(local_max & (binary > 0), 1, 0)

    # Background markers: zeros with distance > threshold
    bg_markers = np.where((binary == 0), 2, 0)

    return fg_markers + bg_markers
