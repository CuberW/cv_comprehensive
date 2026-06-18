"""Instance segmentation demo with local fallback and optional Mask R-CNN."""
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
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn

        _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _WEIGHTS = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        _MODEL = maskrcnn_resnet50_fpn(weights=_WEIGHTS)
        _MODEL.to(_DEVICE)
        _MODEL.eval()
    return _MODEL, _WEIGHTS, _DEVICE


def build_pipeline(image_path=None, score_threshold=0.5, mask_threshold=0.5, **kwargs):
    """Run real Mask R-CNN when explicitly enabled; otherwise run locally."""
    if os.environ.get('CV_ENABLE_PRETRAINED_DEMOS') != '1' or not image_path:
        return _build_local_pipeline(image_path=image_path, **kwargs)

    score_th = float(kwargs.get('score_threshold', kwargs.get('threshold', score_threshold)))
    mask_th = float(kwargs.get('mask_threshold', mask_threshold))
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
    masks = output['masks'].detach().cpu().numpy()[:, 0]
    keep = np.where(scores >= score_th)[0][:20]

    categories = weights.meta.get('categories', [])
    instances = []
    for idx in keep:
        label_idx = int(labels[idx])
        label = categories[label_idx] if label_idx < len(categories) else str(label_idx)
        instances.append({
            'box': [round(float(v), 1) for v in boxes[idx].tolist()],
            'label': label,
            'score': round(float(scores[idx]), 4),
            'mask': (masks[idx] >= mask_th),
        })

    return _package_result(
        img_u8,
        instances,
        status='pretrained_model',
        model='maskrcnn_resnet50_fpn',
        backend='torchvision',
        extra_metrics={
            'weights': str(weights),
            'device': str(device),
            'threshold': score_th,
            'mask_threshold': mask_th,
        },
        explanations=[
            'Image sent to torchvision Mask R-CNN ResNet50-FPN COCO weights.',
            f'Kept detection boxes with score >= {score_th:.2f}.',
            f'Masks come from Mask R-CNN mask head using threshold {mask_th:.2f}.',
            'Each crop shows the original pixels inside one predicted instance.',
        ],
    )


def _build_local_pipeline(image_path=None, **kwargs):
    score_th = float(kwargs.get('score_threshold', kwargs.get('threshold', 0.28)))
    count = int(kwargs.get('num_instances', 4))
    img_u8 = _load_or_fixture(image_path=image_path)
    instances = _local_instances(img_u8, score_threshold=score_th, max_instances=count)
    return _package_result(
        img_u8,
        instances,
        status='local_teaching_fallback',
        model='local_proposal_mask_segmenter',
        backend='NumPy/PIL',
        extra_metrics={
            'pretrained_model': 'maskrcnn_resnet50_fpn',
            'pretrained_enabled': False,
            'threshold': score_th,
        },
        explanations=[
            'Local instance segmentation starts by finding salient object proposals.',
            'Proposal boxes approximate where separate object instances may exist.',
            'Inside each box, color and contrast form a per-instance mask instead of one shared semantic class map.',
            'The strip emphasizes that instance segmentation separates objects one by one.',
        ],
    )


def _local_instances(img, score_threshold=0.28, max_instances=4):
    from app.modules.phase4_deep_learning.detection.processor import _local_candidates

    gray = ensure_gray(img)
    candidates, _ = _local_candidates(gray)
    selected = [d for d in candidates if d['score'] >= score_threshold][:max(1, max_instances)]
    instances = []
    for i, det in enumerate(selected):
        x1, y1, x2, y2 = [int(v) for v in det['box']]
        patch = img[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        center_color = patch.reshape(-1, 3).mean(axis=0)
        dist = np.sqrt(np.sum((patch.astype(np.float32) - center_color[None, None, :]) ** 2, axis=2))
        edge = _edge_norm(ensure_gray(patch))
        color_mask = dist <= max(18.0, np.percentile(dist, 58))
        mask_patch = color_mask | (edge > np.percentile(edge, 72))
        mask = np.zeros(gray.shape, dtype=bool)
        mask[y1:y2, x1:x2] = _clean_mask(mask_patch)
        if mask.sum() < 12:
            mask[y1:y2, x1:x2] = True
        instances.append({
            'box': [x1, y1, x2, y2],
            'label': f'instance {i + 1}',
            'score': det['score'],
            'mask': mask,
        })
    return instances


def _edge_norm(gray):
    g = gray.astype(np.float32)
    gy, gx = np.gradient(g)
    edge = np.sqrt(gx * gx + gy * gy)
    return edge / max(float(edge.max()), 1e-6)


def _clean_mask(mask):
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1, mode='edge')
    votes = np.zeros(mask.shape, dtype=np.uint8)
    for dy in range(3):
        for dx in range(3):
            votes += padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return votes >= 4


def _package_result(img_u8, instances, status, model, backend, extra_metrics, explanations):
    box_vis = _draw_boxes(img_u8, instances)
    mask_vis = _overlay_masks(img_u8, instances)
    detail = _instance_strip(img_u8, instances)
    public_instances = [{k: v for k, v in inst.items() if k != 'mask'} for inst in instances]
    metrics = {
        'status': status,
        'model': model,
        'backend': backend,
        'instances': len(instances),
    }
    metrics.update(extra_metrics)
    return {
        'steps': [
            {'id': 'original', 'name': 'Input image', 'image': img_u8, 'explanation': explanations[0]},
            {'id': 'boxes', 'name': f'Instance boxes ({len(instances)})', 'image': box_vis, 'explanation': explanations[1]},
            {'id': 'masks', 'name': 'Per-instance masks', 'image': mask_vis, 'explanation': explanations[2]},
            {'id': 'detail', 'name': 'Instance crops', 'image': detail, 'explanation': explanations[3]},
        ],
        'metrics': metrics,
        'instances': public_instances,
    }


def _draw_boxes(img, instances):
    vis = img.copy()
    h, w = vis.shape[:2]
    colors = _colors()
    for i, inst in enumerate(instances):
        x1, y1, x2, y2 = [int(round(v)) for v in inst['box']]
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


def _overlay_masks(img, instances):
    out = img.astype(np.float32).copy()
    colors = _colors().astype(np.float32)
    for i, inst in enumerate(instances):
        mask = np.asarray(inst['mask'], dtype=bool)
        if mask.shape[:2] != img.shape[:2]:
            from PIL import Image
            mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((img.shape[1], img.shape[0]), Image.NEAREST)) > 0
        col = colors[i % len(colors)]
        out[mask] = out[mask] * 0.52 + col * 0.48
    return np.clip(out, 0, 255).astype(np.uint8)


def _instance_strip(img, instances):
    from PIL import Image
    if not instances:
        canvas = Image.new('RGB', (420, 120), (15, 23, 42))
        return np.array(canvas)
    thumbs = []
    for inst in instances[:8]:
        x1, y1, x2, y2 = [int(round(v)) for v in inst['box']]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2]
        ratio = 120 / max(1, crop.shape[0])
        pil = Image.fromarray(crop).resize((max(1, int(crop.shape[1] * ratio)), 120), Image.LANCZOS)
        thumbs.append(np.array(pil))
    if not thumbs:
        return np.zeros((120, 240, 3), dtype=np.uint8)
    h = max(t.shape[0] for t in thumbs)
    w = sum(t.shape[1] for t in thumbs) + 6 * (len(thumbs) - 1)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x = 0
    for t in thumbs:
        canvas[:t.shape[0], x:x + t.shape[1]] = t
        x += t.shape[1] + 6
    return canvas


def _colors():
    return np.array([
        [239, 68, 68], [34, 197, 94], [59, 130, 246],
        [217, 119, 6], [124, 58, 237], [20, 184, 166],
    ], dtype=np.uint8)
