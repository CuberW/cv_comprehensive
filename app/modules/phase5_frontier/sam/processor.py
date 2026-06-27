"""Pipeline builder for SAM real prompt segmentation."""
from __future__ import annotations

import numpy as np

from app.utils.image_utils import load_image_u8
from app.modules.phase5_frontier.sam.algorithm import encode_image, predict_from_points


SAM_URL = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'


def build_pipeline(image_path=None, **kwargs):
    if not image_path:
        return {
            'error': 'SAM requires an input image and a local checkpoint.',
            'steps': [],
            'metrics': {'status': 'missing_input'},
        }

    try:
        img_u8 = load_image_u8(image_path, mode='rgb', max_side=768)
    except Exception as exc:
        return {
            'error': f'SAM image load failed: {exc}',
            'steps': [],
            'metrics': {'status': 'bad_input'},
        }

    h, w = img_u8.shape[:2]

    try:
        enc_info = encode_image(img_u8)
    except Exception as exc:
        return _missing_checkpoint(img_u8, exc)

    cx, cy = int(kwargs.get('x', w // 2)), int(kwargs.get('y', h // 2))
    cx, cy = max(0, min(w - 1, cx)), max(0, min(h - 1, cy))
    point_result = predict_from_points(points=[[cx, cy]], labels=[1])

    point_vis = _draw_prompt_point(img_u8, cx, cy)
    best_idx = int(point_result['best_idx'])
    best_mask = point_result['masks'][best_idx]
    mask_overlay = _overlay_mask(img_u8, best_mask, color=(0, 200, 80))
    multi_vis = _make_multi_mask_strip(img_u8, point_result['masks'], point_result['scores'])

    return {
        'steps': [
            {
                'id': 'input',
                'name': '输入图像',
                'image': img_u8,
                'formula': 'I in R^{H x W x 3}',
                'explanation': 'SAM 先对整张图像做一次 ViT 编码。后续不同点或框提示可以复用这份图像 embedding。',
                'data': {'height': h, 'width': w, 'encoder_info': enc_info},
            },
            {
                'id': 'point_prompt',
                'name': '点提示',
                'image': point_vis,
                'formula': 'p = {(x, y, label)}',
                'explanation': f'绿色十字是前景提示点 (x={cx}, y={cy})。Prompt Encoder 将交互提示编码成向量。',
                'data': {'point': [cx, cy], 'label': 1},
            },
            {
                'id': 'best_mask',
                'name': f'最佳 mask (IoU={point_result["scores"][best_idx]:.3f})',
                'image': mask_overlay,
                'formula': 'M = MaskDecoder(E_I, E_p)',
                'explanation': 'Mask Decoder 融合图像 embedding 和提示 embedding，输出候选 mask。绿色区域是真实 SAM 模型预测结果。',
            },
            {
                'id': 'multi_masks',
                'name': '多个候选 mask 对比',
                'image': multi_vis,
                'formula': 'best = argmax_i IoU_i',
                'explanation': 'SAM 会输出多个不同粒度候选，并给出每个候选的 IoU 预测分数。交互式分割常用这些候选让用户选择。',
                'data': {'scores': [round(float(s), 4) for s in point_result['scores']]},
            },
        ],
        'metrics': {
            'status': 'pretrained_model',
            'backend': 'segment-anything',
            'model': 'SAM ViT-B',
            'image_size': f'{w}x{h}',
            'prompt_type': 'point',
            'num_candidates': len(point_result['masks']),
            'best_iou_score': round(float(point_result['scores'][best_idx]), 4),
        },
    }


def _missing_checkpoint(img_u8, exc):
    return {
        'error': (
            'SAM checkpoint is not configured. Download sam_vit_b_01ec64.pth '
            'and either set SAM_CHECKPOINT or place it at models/sam_vit_b_01ec64.pth. '
            f'Official URL: {SAM_URL}. Original error: {exc}'
        ),
        'steps': [
            {
                'id': 'input',
                'name': '输入图像',
                'image': img_u8,
                'formula': 'I in R^{H x W x 3}',
                'explanation': '图像已成功读取；真实 SAM 分割需要本地 ViT-B checkpoint。这里不返回假 mask。',
            },
        ],
        'metrics': {
            'status': 'model_not_available',
            'required_file': 'sam_vit_b_01ec64.pth',
            'env_var': 'SAM_CHECKPOINT',
            'default_path': 'models/sam_vit_b_01ec64.pth',
            'download_url': SAM_URL,
        },
    }


def _draw_prompt_point(img, x, y):
    out = img.copy()
    h, w = out.shape[:2]
    radius = max(6, min(w, h) // 70)
    out[max(0, y - radius):min(h, y + radius + 1), max(0, x - 1):min(w, x + 2)] = [0, 255, 0]
    out[max(0, y - 1):min(h, y + 2), max(0, x - radius):min(w, x + radius + 1)] = [0, 255, 0]
    return out


def _overlay_mask(img, mask, color=(0, 200, 80)):
    from PIL import Image

    h, w = img.shape[:2]
    mask_bool = np.asarray(mask) > 128
    if mask_bool.shape[:2] != (h, w):
        mask_bool = np.array(
            Image.fromarray(mask_bool.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
        ) > 128
    overlay = img.copy().astype(np.float32)
    alpha = 0.45
    for c in range(3):
        overlay[mask_bool, c] = overlay[mask_bool, c] * (1.0 - alpha) + color[c] * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _make_multi_mask_strip(img, masks, scores):
    from PIL import Image, ImageDraw

    thumb_h = 150
    thumbs = []
    for i, mask in enumerate(masks):
        overlay = _overlay_mask(img, mask)
        pil = Image.fromarray(overlay)
        scale = thumb_h / max(1, pil.height)
        thumb = pil.resize((max(1, int(pil.width * scale)), thumb_h), Image.BILINEAR)
        canvas = Image.new('RGB', (thumb.width, thumb_h + 24), (248, 250, 252))
        canvas.paste(thumb, (0, 24))
        ImageDraw.Draw(canvas).text((4, 4), f'mask {i + 1}: IoU {scores[i]:.3f}', fill=(15, 23, 42))
        thumbs.append(canvas)

    total_w = sum(t.width for t in thumbs) + max(0, len(thumbs) - 1) * 6
    out = Image.new('RGB', (total_w, thumb_h + 24), (248, 250, 252))
    x = 0
    for thumb in thumbs:
        out.paste(thumb, (x, 0))
        x += thumb.width + 6
    return np.array(out)
