"""Semantic segmentation demo with local fallback and optional FCN."""
import os

import numpy as np

from app.modules.offline_teaching import _load_or_fixture
from app.utils.image_utils import load_image_u8


_MODEL = None
_WEIGHTS = None
_DEVICE = None


def _get_model():
    global _MODEL, _WEIGHTS, _DEVICE
    if _MODEL is None:
        import torch
        from torchvision.models.segmentation import FCN_ResNet50_Weights, fcn_resnet50

        _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _WEIGHTS = FCN_ResNet50_Weights.DEFAULT
        _MODEL = fcn_resnet50(weights=_WEIGHTS)
        _MODEL.to(_DEVICE)
        _MODEL.eval()
    return _MODEL, _WEIGHTS, _DEVICE


def build_pipeline(image_path=None, **kwargs):
    """Run real FCN when explicitly enabled; otherwise run local segmentation."""
    if os.environ.get('CV_ENABLE_PRETRAINED_DEMOS') != '1' or not image_path:
        return _build_local_pipeline(image_path=image_path, **kwargs)

    img_u8 = load_image_u8(image_path, mode='rgb', max_side=768)
    try:
        model, weights, device = _get_model()
    except Exception as exc:
        fallback = _build_local_pipeline(image_path=image_path, **kwargs)
        fallback['metrics']['pretrained_status'] = 'unavailable'
        fallback['metrics']['pretrained_error'] = f'{type(exc).__name__}: {exc}'
        return fallback

    import torch
    preprocess = weights.transforms()
    tensor = preprocess(_pil_image(img_u8)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)['out'][0]
        probs = torch.softmax(logits, dim=0)
        seg = torch.argmax(probs, dim=0).detach().cpu().numpy().astype(np.int32)
        conf = torch.max(probs, dim=0).values.detach().cpu().numpy()

    categories = weights.meta.get('categories', [])
    seg_resized = _resize_nearest(seg, img_u8.shape[:2])
    conf_resized = _resize_float(conf, img_u8.shape[:2])
    color = _colorize(seg_resized)
    overlay = _overlay(img_u8, color)
    conf_vis = _confidence_heatmap(conf_resized)
    label_summary = _label_summary(seg_resized, categories)
    return {
        'steps': [
            {'id': 'original', 'name': 'Input image', 'image': img_u8,
             'explanation': 'Image sent to torchvision FCN ResNet50 semantic segmentation weights.'},
            {'id': 'segmentation', 'name': 'Real pixel class map', 'image': color,
             'explanation': 'Each pixel color comes from the argmax class after FCN softmax.'},
            {'id': 'confidence', 'name': 'Pixel confidence heatmap', 'image': conf_vis,
             'explanation': 'Brighter pixels have higher maximum class probability.'},
            {'id': 'overlay', 'name': 'Segmentation overlay', 'image': overlay,
             'explanation': 'Predicted semantic classes are blended onto the input image.'},
        ],
        'metrics': {
            'status': 'pretrained_model',
            'model': 'fcn_resnet50',
            'weights': str(weights),
            'backend': 'torchvision',
            'device': str(device),
            'classes_present': len(np.unique(seg_resized)),
            'top_labels': ', '.join(item['label'] for item in label_summary[:4]),
        },
        'labels': label_summary,
    }


def _build_local_pipeline(image_path=None, **kwargs):
    num_classes = int(kwargs.get('num_classes', 5))
    num_classes = int(np.clip(num_classes, 2, 8))
    img_u8 = _load_or_fixture(image_path=image_path)
    seg, conf, centers = _segment_local(img_u8, num_classes=num_classes)
    color = _colorize(seg)
    overlay = _overlay(img_u8, color)
    conf_vis = _confidence_heatmap(conf)
    labels = _local_labels(seg, centers)
    return {
        'steps': [
            {'id': 'input', 'name': 'Input image', 'image': img_u8,
             'explanation': 'Local semantic segmentation treats each pixel as color plus position features.'},
            {'id': 'segmentation', 'name': 'Pixel groups / semantic regions', 'image': color,
             'explanation': 'Pixels assigned to the same region share similar color and nearby spatial location.'},
            {'id': 'confidence', 'name': 'Region confidence', 'image': conf_vis,
             'explanation': 'Confidence is higher when a pixel is clearly closer to its region center than to alternatives.'},
            {'id': 'overlay', 'name': 'Overlay on input', 'image': overlay,
             'explanation': 'The colored mask shows how a dense prediction covers every pixel, unlike object detection boxes.'},
        ],
        'metrics': {
            'status': 'local_teaching_fallback',
            'model': 'local_color_spatial_segmentation',
            'backend': 'NumPy/PIL',
            'pretrained_model': 'fcn_resnet50',
            'pretrained_enabled': False,
            'classes_present': len(labels),
            'num_classes': num_classes,
        },
        'labels': labels,
    }


def _segment_local(img, num_classes=5):
    arr = np.asarray(img, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    xy = np.stack([yy / max(h - 1, 1), xx / max(w - 1, 1)], axis=-1)
    feats = np.concatenate([arr, xy * 0.45], axis=-1).reshape(-1, 5)
    rng = np.random.default_rng(2026)
    picks = np.linspace(0, len(feats) - 1, num_classes).astype(int)
    jitter = rng.integers(0, max(1, len(feats) // num_classes), size=num_classes)
    centers = feats[np.clip(picks + jitter, 0, len(feats) - 1)].copy()
    for _ in range(9):
        dists = np.sum((feats[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        for k in range(num_classes):
            if np.any(labels == k):
                centers[k] = feats[labels == k].mean(axis=0)
    dists = np.sum((feats[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    order = np.sort(dists, axis=1)
    labels = np.argmin(dists, axis=1).reshape(h, w)
    margin = np.clip((order[:, 1] - order[:, 0]) / (order[:, 1] + 1e-6), 0, 1).reshape(h, w)
    return labels.astype(np.int32), margin.astype(np.float32), centers


def _local_labels(seg, centers):
    labels, counts = np.unique(seg, return_counts=True)
    out = []
    for label, count in sorted(zip(labels, counts), key=lambda x: x[1], reverse=True):
        rgb = centers[int(label), :3]
        name = _color_name(rgb)
        out.append({'label_id': int(label), 'label': name, 'pixels': int(count)})
    return out


def _label_summary(seg, categories):
    labels, counts = np.unique(seg, return_counts=True)
    top = sorted(zip(labels.tolist(), counts.tolist()), key=lambda x: x[1], reverse=True)[:6]
    return [
        {
            'label_id': int(label),
            'label': categories[int(label)] if int(label) < len(categories) else str(label),
            'pixels': int(count),
        }
        for label, count in top
    ]


def _color_name(rgb):
    r, g, b = rgb
    if max(rgb) - min(rgb) < 0.12:
        return 'neutral region'
    if r >= g and r >= b:
        return 'warm/red region'
    if g >= r and g >= b:
        return 'green region'
    return 'blue/cool region'


def _pil_image(arr):
    from PIL import Image
    return Image.fromarray(arr.astype(np.uint8))


def _resize_nearest(arr, shape_hw):
    from PIL import Image
    h, w = shape_hw
    return np.array(Image.fromarray(arr.astype(np.int32), mode='I').resize((w, h), Image.NEAREST), dtype=np.int32)


def _resize_float(arr, shape_hw):
    from PIL import Image
    h, w = shape_hw
    return np.array(Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0


def _colorize(seg):
    palette = np.array([
        [20, 23, 30], [239, 68, 68], [34, 197, 94], [59, 130, 246],
        [245, 158, 11], [168, 85, 247], [20, 184, 166], [236, 72, 153],
        [250, 204, 21],
    ], dtype=np.uint8)
    return palette[np.asarray(seg, dtype=np.int32) % len(palette)]


def _overlay(img, color):
    return np.clip(img.astype(np.float32) * 0.55 + color.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)


def _confidence_heatmap(conf):
    conf = np.clip(np.asarray(conf, dtype=np.float32), 0, 1)
    r = (conf * 255).astype(np.uint8)
    g = ((1 - np.abs(conf - 0.5) * 2) * 180).astype(np.uint8)
    b = ((1 - conf) * 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)
