"""Simplified grid-based object detection (YOLO-style demo). Pure NumPy."""
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def compute_iou(box1, box2):
    """
    Compute IoU (Intersection over Union) between two boxes.
    Each box: [x_center, y_center, width, height] normalized [0,1].
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to corner coordinates
    left1, top1 = x1 - w1/2, y1 - h1/2
    right1, bottom1 = x1 + w1/2, y1 + h1/2
    left2, top2 = x2 - w2/2, y2 - h2/2
    right2, bottom2 = x2 + w2/2, y2 + h2/2

    # Intersection area
    inter_left = max(left1, left2)
    inter_top = max(top1, top2)
    inter_right = min(right1, right2)
    inter_bottom = min(bottom1, bottom2)

    if inter_right <= inter_left or inter_bottom <= inter_top:
        return 0.0

    inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter_area

    return float(inter_area / union) if union > 0 else 0.0


def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """
    NMS: remove overlapping boxes, keeping the highest-scoring one.
    boxes: [[x, y, w, h], ...] normalized
    scores: [score, ...]
    Returns: indices of kept boxes
    """
    if len(boxes) == 0:
        return []

    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0:
        idx = order[0]
        keep.append(int(idx))
        if len(order) == 1:
            break

        ious = np.array([compute_iou(boxes[idx], boxes[j]) for j in order[1:]])
        order = order[1:][ious < iou_threshold]

    return keep


def grid_detection_demo(img_shape, grid_size=7, num_boxes=2, num_classes=3):
    """
    Generate a demo detection result on a grid.
    Simulates YOLO-style output: grid cells each predict bounding boxes.

    For a real implementation, a CNN would produce the grid predictions.
    This demo generates plausible detections for visualization purposes.

    Returns: detected boxes, class labels, confidence scores
    """
    h, w = img_shape[:2]
    cell_h, cell_w = h // grid_size, w // grid_size

    rng = np.random.default_rng(42)
    detections = []

    for gy in range(grid_size):
        for gx in range(grid_size):
            for b in range(num_boxes):
                # Simulate: most cells have low confidence, a few have high
                confidence = float(rng.random()) ** 5  # Skew to low values

                if confidence < 0.3:
                    continue

                # Box center relative to cell, then absolute
                bx = (gx + rng.random()) / grid_size
                by = (gy + rng.random()) / grid_size
                bw = rng.random() * 0.4 + 0.1
                bh = rng.random() * 0.4 + 0.1

                # Class prediction
                class_probs = rng.random(num_classes)
                class_probs = class_probs / class_probs.sum()
                class_id = int(np.argmax(class_probs))

                detections.append({
                    'box': [float(bx), float(by), float(bw), float(bh)],
                    'confidence': float(confidence),
                    'class_id': class_id,
                })

    # Sort by confidence
    detections.sort(key=lambda d: d['confidence'], reverse=True)

    # NMS
    boxes = [d['box'] for d in detections]
    scores = [d['confidence'] for d in detections]
    keep_idx = non_max_suppression(boxes, scores, iou_threshold=0.4)

    result = []
    for idx in keep_idx[:20]:  # Max 20 detections
        d = detections[idx]
        box = d['box']
        result.append({
            'x': round(float(box[0] * w), 1),
            'y': round(float(box[1] * h), 1),
            'w': round(float(box[2] * w), 1),
            'h': round(float(box[3] * h), 1),
            'confidence': round(float(d['confidence']), 3),
            'class_id': int(d['class_id']),
        })

    return result


def compute_map(predictions, ground_truth, iou_threshold=0.5, num_classes=3):
    """
    Compute simplified mAP (mean Average Precision).
    predictions: list of {box, confidence, class_id}
    ground_truth: list of {box, class_id}
    Returns: mAP value and per-class APs
    """
    aps = []
    for c in range(num_classes):
        preds_c = [(p['confidence'], p['box']) for p in predictions if p['class_id'] == c]
        gts_c = [g['box'] for g in ground_truth if g['class_id'] == c]

        if len(preds_c) == 0 or len(gts_c) == 0:
            aps.append(0.0)
            continue

        preds_c.sort(reverse=True)
        tp = np.zeros(len(preds_c))
        fp = np.zeros(len(preds_c))
        gt_used = [False] * len(gts_c)

        for i, (_, box) in enumerate(preds_c):
            best_iou = 0
            best_j = -1
            for j, gt_box in enumerate(gts_c):
                if gt_used[j]:
                    continue
                iou = compute_iou(box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= iou_threshold:
                tp[i] = 1
                gt_used[best_j] = True
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / len(gts_c)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-12)

        # 11-point interpolated AP
        ap = 0.0
        for r in np.linspace(0, 1, 11):
            if (recalls >= r).any():
                ap += float(precisions[recalls >= r].max())
        aps.append(ap / 11.0)

    return float(np.mean(aps)) if aps else 0.0, [float(a) for a in aps]
