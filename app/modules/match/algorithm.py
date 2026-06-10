"""
Feature matching + homography estimation + image stitching.
Pure NumPy: descriptor distance -> ratio test -> RANSAC -> homography -> stitching.
"""
import numpy as np
import imageio.v3 as iio
from app.modules.corner.algorithm import harris_pipeline
from app.modules.sift.algorithm import sift_pipeline, ensure_uint8


def resize_max_side(img, max_side=420):
    """Resize so longest side <= max_side."""
    arr = ensure_uint8(img)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.shape[2] == 4:
        arr = arr[..., :3]
    h, w = arr.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale >= 0.999:
        return arr, 1.0
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    ys = np.linspace(0, h - 1, nh).astype(np.int32)
    xs = np.linspace(0, w - 1, nw).astype(np.int32)
    return arr[ys[:, None], xs[None, :]], scale


def normalize_descriptor(vec):
    """L2 normalize a descriptor vector."""
    arr = np.asarray(vec, dtype=np.float32).ravel()
    arr = arr - float(arr.mean())
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 1e-12 else arr * 0.0


def extract_features(img, method='sift', max_points=260):
    """
    Unified feature extraction interface.
    method='sift': 128-dim SIFT descriptors
    method='harris': Harris corners + 81-dim patch descriptors
    Returns: (rgb_img, points_list, descriptors_array, scale)
    """
    rgb, scale = resize_max_side(img)

    if method == 'harris':
        data = harris_pipeline(rgb, threshold_ratio=0.01, window_size=3,
                               sigma=2.0, max_points=max_points)
        gray = data['gray'].astype(np.float32) / 255.0
        descs, points, r = [], [], 4
        for p in data['points'][:max_points]:
            x, y = int(round(p['x'])), int(round(p['y']))
            patch = np.zeros((2*r+1, 2*r+1), dtype=np.float32)
            for yy in range(-r, r+1):
                sy = int(np.clip(y+yy, 0, gray.shape[0]-1))
                for xx in range(-r, r+1):
                    sx = int(np.clip(x+xx, 0, gray.shape[1]-1))
                    patch[yy+r, xx+r] = gray[sy, sx]
            descs.append(normalize_descriptor(patch))
            points.append({'x': float(x), 'y': float(y), 'scale': 3.0,
                           'orientation': 0.0, 'response': float(p.get('response', 0.0))})
        return rgb, points, np.asarray(descs, dtype=np.float32), scale

    # SIFT
    data = sift_pipeline(rgb, octaves=3, num_layers=4, threshold=0.015,
                         max_keypoints=max_points, max_compute_side=420)
    points = [{'x': float(kp['x']), 'y': float(kp['y']),
               'scale': float(kp.get('scale', 0.0)),
               'orientation': float(kp.get('orientation', 0.0)),
               'response': float(kp.get('response', 0.0))}
              for kp in data['keypoints'][:len(data['descriptors'])]]
    descs = np.asarray(data['descriptors'], dtype=np.float32)
    if descs.ndim != 2:
        descs = np.zeros((0, 128), dtype=np.float32)
    return rgb, points, descs, scale


def pairwise_distances(a, b):
    """Euclidean distance matrix: dist[i,j] = ||a[i] - b[j]||.
    Vectorized: ||a-b||^2 = a^2 + b^2 - 2ab^T"""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    aa = np.sum(a * a, axis=1)[:, None]
    bb = np.sum(b * b, axis=1)[None, :]
    d2 = np.maximum(aa + bb - 2.0 * a @ b.T, 0.0)
    return np.sqrt(d2).astype(np.float32)


def match_descriptors(desc1, desc2, ratio=0.8, max_matches=180):
    """
    Nearest neighbor + Lowe's ratio test.
    For each query descriptor, find the nearest and second-nearest neighbor.
    Accept match if d1/d2 < ratio (nearest is significantly better than second-nearest).
    Typical ratio=0.8. Lower ratio -> fewer false matches but may miss correct ones.
    """
    if len(desc1) == 0 or len(desc2) < 2:
        return [], []
    dists = pairwise_distances(desc1, desc2)
    raw, accepted = [], []
    for i in range(dists.shape[0]):
        order = np.argsort(dists[i])[:2]
        first, second = int(order[0]), int(order[1])
        d1, d2 = float(dists[i, first]), float(dists[i, second])
        r = d1 / d2 if d2 > 1e-12 else 1.0
        item = {'left': i, 'right': first, 'second': second, 'd1': d1, 'd2': d2, 'ratio': r}
        raw.append(item)
        if r < ratio:
            accepted.append(item)
    accepted.sort(key=lambda m: (m['ratio'], m['d1']))
    raw.sort(key=lambda m: (m['ratio'], m['d1']))
    return raw[:max_matches], accepted[:max_matches]


def homography_from_points(src, dst):
    """
    Estimate homography H (3x3) from >=4 point pairs using DLT.
    Constraint: dst = H @ src (homogeneous coords).
    Solve Ah = 0 via SVD (last column of V).
    """
    if len(src) < 4:
        return None
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    A = np.asarray(A, dtype=np.float64)
    try:
        _, _, vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    H = vt[-1].reshape(3, 3)
    if abs(H[2, 2]) < 1e-12:
        return None
    return H / H[2, 2]


def apply_homography(points, H):
    """Apply homography transformation to point set."""
    pts = np.asarray(points, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    hp = np.hstack([pts, ones]) @ H.T
    denom = hp[:, 2:3]
    denom[np.abs(denom) < 1e-12] = 1e-12
    return hp[:, :2] / denom


def ransac_homography(src, dst, iterations=500, threshold=4.0):
    """
    RANSAC robust homography estimation.
    Each iteration: randomly sample 4 matches, fit H, count inliers.
    Final: re-estimate H using all inliers.
    """
    if len(src) < 4:
        return np.eye(3, dtype=np.float64), []

    src, dst = np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64)
    rng = np.random.default_rng(20260609)
    best_inliers = np.zeros(len(src), dtype=bool)
    best_H = None

    for _ in range(iterations):
        idx = rng.choice(len(src), 4, replace=False)
        H = homography_from_points(src[idx], dst[idx])
        if H is None:
            continue
        pred = apply_homography(src, H)
        err = np.linalg.norm(pred - dst, axis=1)
        inliers = err < threshold
        if int(inliers.sum()) > int(best_inliers.sum()):
            best_inliers = inliers
            best_H = H

    if best_H is None:
        fallback = homography_from_points(src, dst)
        best_H = fallback if fallback is not None else np.eye(3, dtype=np.float64)

    if int(best_inliers.sum()) >= 4:
        refined = homography_from_points(src[best_inliers], dst[best_inliers])
        if refined is not None:
            best_H = refined

    return best_H, np.where(best_inliers)[0].astype(int).tolist()


def draw_matches(left_img, right_img, pts1, pts2, matches, inlier_set):
    """Draw side-by-side images with match lines. Green=inlier, Red=outlier."""
    h = max(left_img.shape[0], right_img.shape[0])
    w = left_img.shape[1] + right_img.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:left_img.shape[0], :left_img.shape[1]] = left_img
    canvas[:right_img.shape[0], left_img.shape[1]:] = right_img
    for idx, m in enumerate(matches[:80]):
        color = np.array([34, 197, 94] if idx in inlier_set else [239, 68, 68], dtype=np.uint8)
        x1, y1 = int(round(pts1[m['left']]['x'])), int(round(pts1[m['left']]['y']))
        x2, y2 = int(round(pts2[m['right']]['x'] + left_img.shape[1])), int(round(pts2[m['right']]['y']))
        n = max(abs(x2-x1), abs(y2-y1), 1)
        for t in np.linspace(0, 1, n):
            x, y = int(round(x1+(x2-x1)*t)), int(round(y1+(y2-y1)*t))
            if 0 <= y < h and 0 <= x < w:
                canvas[y, x] = color
    return canvas


def stitch_images(left, right, H, max_output_side=1200):
    """
    Stitch right image into left image coordinate system using homography H.
    Uses inverse mapping + bilinear interpolation for hole-free result.
    """
    if H is None:
        canvas = left.copy()
        if canvas.ndim == 2:
            canvas = np.repeat(canvas[..., None], 3, axis=2)
        return canvas

    h1, w1 = left.shape[:2]
    h2, w2 = right.shape[:2]

    right_corners = np.array([[0,0],[w2-1,0],[w2-1,h2-1],[0,h2-1]], dtype=np.float64)
    left_corners = np.array([[0,0],[w1-1,0],[w1-1,h1-1],[0,h1-1]], dtype=np.float64)
    warped = apply_homography(right_corners, H)
    all_corners = np.vstack([left_corners, warped])
    min_xy = np.floor(all_corners.min(axis=0)).astype(np.int32)
    max_xy = np.ceil(all_corners.max(axis=0)).astype(np.int32)
    out_w, out_h = int(max_xy[0]-min_xy[0]+1), int(max_xy[1]-min_xy[1]+1)

    if out_w < 1 or out_h < 1 or out_w > 2400 or out_h > 1600:
        H = np.eye(3, dtype=np.float64)
        warped = apply_homography(right_corners, H)
        all_corners = np.vstack([left_corners, warped])
        min_xy = np.floor(all_corners.min(axis=0)).astype(np.int32)
        max_xy = np.ceil(all_corners.max(axis=0)).astype(np.int32)
        out_w, out_h = int(max_xy[0]-min_xy[0]+1), int(max_xy[1]-min_xy[1]+1)

    offset = -min_xy
    canvas = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weight = np.zeros((out_h, out_w, 1), dtype=np.float32)

    # Place left image
    lx, ly = int(offset[0]), int(offset[1])
    canvas[ly:ly+h1, lx:lx+w1] += left.astype(np.float32)
    weight[ly:ly+h1, lx:lx+w1] += 1.0

    # Inverse map right image
    try:
        invH = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        invH = np.eye(3, dtype=np.float64)

    yy, xx = np.indices((out_h, out_w), dtype=np.float64)
    world_x = xx.ravel() - offset[0]
    world_y = yy.ravel() - offset[1]
    pts = np.stack([world_x, world_y, np.ones_like(world_x)], axis=1) @ invH.T
    denom = pts[:, 2]
    valid_denom = np.abs(denom) > 1e-12
    rx, ry = np.zeros_like(world_x), np.zeros_like(world_y)
    rx[valid_denom] = pts[valid_denom, 0] / denom[valid_denom]
    ry[valid_denom] = pts[valid_denom, 1] / denom[valid_denom]
    valid = valid_denom & (rx >= 0) & (rx <= w2-1) & (ry >= 0) & (ry <= h2-1)

    if np.any(valid):
        x0, y0 = np.floor(rx[valid]).astype(np.int32), np.floor(ry[valid]).astype(np.int32)
        x1, y1 = np.clip(x0+1, 0, w2-1), np.clip(y0+1, 0, h2-1)
        x0, y0 = np.clip(x0, 0, w2-1), np.clip(y0, 0, h2-1)
        wx = (rx[valid] - x0).reshape(-1, 1)
        wy = (ry[valid] - y0).reshape(-1, 1)
        top = right[y0, x0].astype(np.float32)*(1-wx) + right[y0, x1].astype(np.float32)*wx
        bottom = right[y1, x0].astype(np.float32)*(1-wx) + right[y1, x1].astype(np.float32)*wx
        sampled = top*(1-wy) + bottom*wy

        flat_canvas = canvas.reshape(-1, 3)
        flat_weight = weight.reshape(-1, 1)
        flat_canvas[valid] += sampled
        flat_weight[valid] += 1.0

    result = canvas / np.maximum(weight, 1.0)
    result[weight[..., 0] <= 0] = 18
    result = np.clip(result, 0, 255).astype(np.uint8)

    h_out, w_out = result.shape[:2]
    if max(h_out, w_out) > int(max_output_side):
        s = float(max_output_side) / float(max(h_out, w_out))
        nh, nw = max(1, int(round(h_out*s))), max(1, int(round(w_out*s)))
        ys = np.linspace(0, h_out-1, nh).astype(np.int32)
        xs = np.linspace(0, w_out-1, nw).astype(np.int32)
        result = result[ys[:, None], xs[None, :]]

    return result


def build_match_pipeline(left_path, right_path, method='sift', ratio=0.8):
    """Build complete feature matching pipeline data."""
    left_img = ensure_uint8(iio.imread(left_path) if isinstance(left_path, str) else left_path)
    right_img = ensure_uint8(iio.imread(right_path) if isinstance(right_path, str) else right_path)

    left, pts1, desc1, _ = extract_features(left_img, method=method)
    right, pts2, desc2, _ = extract_features(right_img, method=method)

    raw, matches = match_descriptors(desc1, desc2, ratio=ratio)

    src = np.array([[pts2[m['right']]['x'], pts2[m['right']]['y']] for m in matches], dtype=np.float64)
    dst = np.array([[pts1[m['left']]['x'], pts1[m['left']]['y']] for m in matches], dtype=np.float64)
    H, inliers = ransac_homography(src, dst)
    inlier_set = set(inliers)

    stitched = stitch_images(left, right, H)
    match_preview = draw_matches(left, right, pts1, pts2, matches, inlier_set)

    match_data = [{
        'left': int(m['left']), 'right': int(m['right']), 'second': int(m['second']),
        'd1': round(float(m['d1']), 5), 'd2': round(float(m['d2']), 5),
        'ratio': round(float(m['ratio']), 5), 'inlier': i in inlier_set,
    } for i, m in enumerate(matches[:120])]

    return {
        'left': left, 'right': right, 'pts1': pts1, 'pts2': pts2,
        'match_preview': match_preview, 'stitched': stitched,
        'matches': match_data,
        'raw_count': int(len(raw)), 'match_count': int(len(matches)),
        'inlier_count': int(len(inliers)),
        'H': np.round(H, 6).tolist() if H is not None else None,
    }
