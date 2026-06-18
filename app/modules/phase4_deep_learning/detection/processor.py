"""Object detection demo with local fallback and optional Faster R-CNN."""
import os

import numpy as np

from app.modules.offline_teaching import _load_or_fixture
from app.utils.image_utils import ensure_gray, load_image_u8


_MODEL = None
_WEIGHTS = None
_DEVICE = None


def _get_model():
    global _MODEL, _WEIGHTS, _DEVICE
    if _MODEL is None:
        import torch
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_Weights,
            fasterrcnn_resnet50_fpn,
        )

        _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _WEIGHTS = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        _MODEL = fasterrcnn_resnet50_fpn(weights=_WEIGHTS)
        _MODEL.to(_DEVICE)
        _MODEL.eval()
    return _MODEL, _WEIGHTS, _DEVICE


def build_pipeline(image_path=None, score_threshold=0.5, **kwargs):
    """Run real Faster R-CNN when explicitly enabled; otherwise run locally."""
    if os.environ.get('CV_ENABLE_PRETRAINED_DEMOS') != '1' or not image_path:
        return _build_local_pipeline(image_path=image_path, **kwargs)

    threshold = float(kwargs.get('score_threshold', kwargs.get('threshold', score_threshold)))
    img_u8 = load_image_u8(image_path, mode='rgb', max_side=768)
    try:
        model, weights, device = _get_model()
    except Exception as exc:
        fallback = _build_local_pipeline(image_path=image_path, **kwargs)
        fallback['metrics']['pretrained_status'] = 'unavailable'
        fallback['metrics']['pretrained_error'] = f'{type(exc).__name__}: {exc}'
        return fallback

    import torch
    from torchvision.transforms.functional import to_tensor

    tensor = to_tensor(img_u8).to(device)
    with torch.no_grad():
        output = model([tensor])[0]

    scores = output['scores'].detach().cpu().numpy()
    boxes = output['boxes'].detach().cpu().numpy()
    labels = output['labels'].detach().cpu().numpy()
    keep = np.where(scores >= threshold)[0]

    categories = weights.meta.get('categories', [])
    detections = []
    for idx in keep[:30]:
        label_idx = int(labels[idx])
        label = categories[label_idx] if label_idx < len(categories) else str(label_idx)
        detections.append({
            'box': [round(float(v), 1) for v in boxes[idx].tolist()],
            'label': label,
            'score': round(float(scores[idx]), 4),
        })

    result_vis = _draw_boxes(img_u8, detections)
    score_chart = _score_chart(detections)
    return {
        'steps': [
            {'id': 'original', 'name': 'Input image', 'image': img_u8,
             'explanation': 'Image sent to torchvision Faster R-CNN ResNet50-FPN COCO weights.'},
            {'id': 'detections', 'name': f'Real detections ({len(detections)})', 'image': result_vis,
             'explanation': f'Kept model boxes with score >= {threshold:.2f}; no synthetic boxes are injected.'},
            {'id': 'scores', 'name': 'Detection confidence', 'image': score_chart,
             'explanation': 'Confidence chart sorted by model output score.'},
        ],
        'metrics': {
            'status': 'pretrained_model',
            'model': 'fasterrcnn_resnet50_fpn',
            'weights': str(weights),
            'backend': 'torchvision',
            'device': str(device),
            'threshold': threshold,
            'detections': len(detections),
        },
        'detections': detections,
    }


def _build_local_pipeline(image_path=None, **kwargs):
    threshold = float(kwargs.get('score_threshold', kwargs.get('threshold', 0.32)))
    img_u8 = _load_or_fixture(image_path=image_path)
    gray = ensure_gray(img_u8)
    candidates, score_map = _local_candidates(gray)
    detections = [d for d in candidates if d['score'] >= threshold][:12]
    result_vis = _draw_boxes(img_u8, detections)
    score_chart = _score_chart(detections)
    heat = _heatmap(score_map)
    return {
        'steps': [
            {'id': 'input', 'name': 'Input image', 'image': img_u8,
             'explanation': 'Local detection demo starts from image contrast, edges, and salient regions.'},
            {'id': 'objectness', 'name': 'Objectness heatmap', 'image': heat,
             'explanation': 'Brighter areas have stronger edges or local contrast and become candidate object regions.'},
            {'id': 'detections', 'name': f'Candidate boxes after NMS ({len(detections)})', 'image': result_vis,
             'explanation': 'Candidates are sorted by score and filtered with non-maximum suppression.'},
            {'id': 'scores', 'name': 'Candidate confidence', 'image': score_chart,
             'explanation': 'This is an offline teaching detector. Enable pretrained demos to run real Faster R-CNN.'},
        ],
        'metrics': {
            'status': 'local_teaching_fallback',
            'model': 'local_edge_objectness_detector',
            'backend': 'NumPy/PIL',
            'pretrained_model': 'fasterrcnn_resnet50_fpn',
            'pretrained_enabled': False,
            'threshold': threshold,
            'detections': len(detections),
        },
        'detections': detections,
    }


def _local_candidates(gray):
    g = gray.astype(np.float32)
    gy, gx = np.gradient(g)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = mag / max(float(mag.max()), 1e-6)
    h, w = gray.shape
    cell = max(18, min(h, w) // 5)
    stride = max(8, cell // 2)
    boxes = []
    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            y1 = min(h, y0 + cell)
            x1 = min(w, x0 + cell)
            if y1 - y0 < 12 or x1 - x0 < 12:
                continue
            edge_score = float(mag[y0:y1, x0:x1].mean())
            contrast = float(g[y0:y1, x0:x1].std() / 96.0)
            score = float(np.clip(edge_score * 1.8 + contrast * 0.5, 0, 1))
            if score >= 0.18:
                boxes.append({'box': [x0, y0, x1, y1], 'label': 'salient region', 'score': round(score, 4)})
    boxes.sort(key=lambda d: d['score'], reverse=True)
    keep = []
    for det in boxes:
        if all(_box_iou(det['box'], old['box']) < 0.35 for old in keep):
            keep.append(det)
    return keep, mag


def _box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def _draw_boxes(img, detections):
    vis = img.copy()
    h, w = vis.shape[:2]
    colors = np.array([
        [239, 68, 68], [34, 197, 94], [59, 130, 246],
        [217, 119, 6], [124, 58, 237], [20, 184, 166],
    ], dtype=np.uint8)
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = [int(round(v)) for v in det['box']]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        col = colors[i % len(colors)]
        vis[y1:y1 + 3, x1:x2] = col
        vis[max(y2 - 2, y1):y2 + 1, x1:x2] = col
        vis[y1:y2, x1:x1 + 3] = col
        vis[y1:y2, max(x2 - 2, x1):x2 + 1] = col
    return vis


def _score_chart(detections, width=640, height=320):
    from PIL import Image, ImageDraw
    canvas = Image.new('RGB', (width, height), (15, 23, 42))
    draw = ImageDraw.Draw(canvas)
    if not detections:
        draw.text((20, height // 2 - 8), 'No detections above threshold', fill=(226, 232, 240))
        return np.array(canvas)
    rows = detections[:10]
    bar_h = max(18, (height - 40) // len(rows) - 6)
    for i, det in enumerate(rows):
        y = 20 + i * (bar_h + 6)
        score = float(det['score'])
        bar_w = int(score * (width - 220))
        draw.rectangle((180, y, 180 + bar_w, y + bar_h), fill=(59, 130, 246))
        draw.text((12, y + 2), f"{det['label']} {score:.2f}", fill=(226, 232, 240))
    return np.array(canvas)


def _heatmap(score):
    s = np.clip(np.asarray(score, dtype=np.float32), 0, 1)
    return np.stack([
        (s * 255).astype(np.uint8),
        ((1 - np.abs(s - 0.5) * 2) * 210).astype(np.uint8),
        ((1 - s) * 255).astype(np.uint8),
    ], axis=-1)
