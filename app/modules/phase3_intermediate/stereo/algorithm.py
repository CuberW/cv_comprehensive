"""Stereo block matching for disparity estimation. Pure NumPy."""
import numpy as np


def block_matching_disparity(left_img, right_img, block_size=9, max_disparity=64):
    """
    Compute disparity map using block matching (SAD - Sum of Absolute Differences).
    For each pixel in the left image, find the best matching patch in the right image
    along the same row (epipolar constraint).

    disparity = x_left - x_right  (larger = closer object)

    Returns: disparity map (float, same size as input)
    """
    left = np.asarray(left_img, dtype=np.float64)
    right = np.asarray(right_img, dtype=np.float64)

    if left.ndim == 3:
        left = 0.299*left[:,:,0] + 0.587*left[:,:,1] + 0.114*left[:,:,2]
    if right.ndim == 3:
        right = 0.299*right[:,:,0] + 0.587*right[:,:,1] + 0.114*right[:,:,2]

    h, w = left.shape
    half = block_size // 2
    disparity = np.zeros((h, w), dtype=np.float64)

    for y in range(half, h - half):
        for x in range(half, w - half):
            patch_left = left[y-half:y+half+1, x-half:x+half+1]
            best_sad = np.inf
            best_d = 0

            # Search range: from x-max_disparity to x
            search_start = max(half, x - max_disparity)
            for d in range(search_start, x - half + 1):
                patch_right = right[y-half:y+half+1, d-half:d+half+1]
                sad = np.sum(np.abs(patch_left - patch_right))
                if sad < best_sad:
                    best_sad = sad
                    best_d = x - d

            disparity[y, x] = best_d

    return disparity


def disparity_to_depth(disparity, baseline=0.1, focal_length=500):
    """
    Convert disparity to depth.
    depth = (baseline * focal_length) / disparity
    depth = 0 where disparity = 0 (no match found)
    """
    disp = np.asarray(disparity, dtype=np.float64)
    depth = np.zeros_like(disp)
    valid = disp > 0
    depth[valid] = (baseline * focal_length) / disp[valid]
    return depth


def disparity_to_color(disparity, max_disp=None):
    """
    Colorize disparity map for visualization.
    Red = close (large disparity), Blue = far (small disparity).
    """
    disp = np.asarray(disparity, dtype=np.float64)
    if max_disp is None:
        max_disp = disp.max() if disp.max() > 0 else 1.0

    h, w = disp.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    norm = np.clip(disp / max(max_disp, 1e-6), 0, 1)
    # Jet colormap: blue (0) -> cyan -> green -> yellow -> red (1)
    # Simplified 3-point mapping
    for y in range(h):
        for x in range(w):
            v = norm[y, x]
            if v < 0.25:
                r, g, b = 0, int(v*4*255), 255
            elif v < 0.5:
                r, g, b = 0, 255, int((0.5-v)*4*255)
            elif v < 0.75:
                r, g, b = int((v-0.5)*4*255), 255, 0
            else:
                r, g, b = 255, int((1-v)*4*255), 0
            colored[y, x] = [r, g, b]

    return colored
