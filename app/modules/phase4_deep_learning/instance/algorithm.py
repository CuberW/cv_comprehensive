"""Simplified instance segmentation (Mask R-CNN style demo). Pure NumPy."""
import numpy as np


def simple_instance_demo(img, num_instances=3):
    """
    Demo instance segmentation: detect + segment individual objects.

    This generates plausible instance masks for visualization.
    A real Mask R-CNN has: Backbone -> RPN -> ROI Align -> Box Head + Mask Head.
    This demo simulates the output: bounding boxes with per-instance masks.

    Returns: list of {box: [x,y,w,h], mask: 2D binary array, class_id: int}
    """
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    h, w = arr.shape[:2]
    rng = np.random.default_rng(2026)

    instances = []
    for i in range(num_instances):
        # Generate a plausible bounding box (not overlapping too much)
        cx = rng.integers(w//6, 5*w//6)
        cy = rng.integers(h//6, 5*h//6)
        bw = rng.integers(w//8, w//3)
        bh = rng.integers(h//8, h//3)

        x1 = max(0, cx - bw//2)
        y1 = max(0, cy - bh//2)
        x2 = min(w, cx + bw//2)
        y2 = min(h, cy + bh//2)

        # Create elliptical mask within the box
        mask = np.zeros((h, w), dtype=np.uint8)
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        yy, xx = np.ogrid[y1:y2, x1:x2]
        dist_y = (yy - (y1 + y2)/2) / (bbox_h/2 + 1e-6)
        dist_x = (xx - (x1 + x2)/2) / (bbox_w/2 + 1e-6)
        ellipse = (dist_y**2 + dist_x**2) <= 0.9  # Slightly smaller than box
        mask[y1:y2, x1:x2] = ellipse.astype(np.uint8) * 255

        # Add some noise to mask boundary for realism
        noise_mask = rng.random((h, w)) > 0.92
        mask = np.clip(mask.astype(np.int32) + noise_mask.astype(np.int32) * 50, 0, 255).astype(np.uint8)

        instances.append({
            'class_id': i % 3,
            'box': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
            'mask': mask,
        })

    return instances


def compute_mask_iou(mask1, mask2):
    """Compute IoU between two binary masks."""
    m1 = np.asarray(mask1, dtype=bool)
    m2 = np.asarray(mask2, dtype=bool)
    intersection = (m1 & m2).sum()
    union = (m1 | m2).sum()
    return float(intersection / union) if union > 0 else 0.0


def colorize_instances(instances, img_shape):
    """Overlay colored instance masks on a black canvas."""
    colors = [
        (220, 38, 38, 128),   # Red
        (37, 99, 235, 128),   # Blue
        (5, 150, 105, 128),   # Green
        (217, 119, 6, 128),   # Orange
        (124, 58, 237, 128),  # Purple
    ]
    h, w = img_shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for idx, inst in enumerate(instances):
        color = colors[idx % len(colors)]
        mask = inst['mask'] > 0
        for c in range(3):
            canvas[:, :, c] = np.where(mask, color[c], canvas[:, :, c])

    return canvas
