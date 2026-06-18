"""Pipeline builder for SAM module."""
import numpy as np
from app.utils.image_utils import load_image_u8, to_base64
from app.modules.phase5_frontier.sam.algorithm import (
    encode_image, predict_from_points, predict_from_box,
)


def build_pipeline(image_path=None, **kwargs):
    if not image_path:
        return {
            'error': 'SAM demo requires an uploaded image.',
            'steps': [],
            'metrics': {'status': 'missing_input'},
        }

    try:
        img_u8 = load_image_u8(image_path, mode='rgb', max_side=768)
    except Exception as exc:
        return {
            'error': f'SAM image load failed: {exc}',
            'steps': [],
            'metrics': {'status': 'model_not_available'},
        }

    h, w = img_u8.shape[:2]

    # Encode image (this is the heavy part - SAM's ViT encoder)
    try:
        enc_info = encode_image(img_u8)
    except Exception as e:
        # If SAM model not available, fallback
        return {
            'error': f'SAM model not available: {e}',
            'steps': [],
            'metrics': {'status': 'model_not_available'},
        }

    # Demo: predict from a center point
    cx, cy = w // 2, h // 2
    point_result = predict_from_points(
        points=[[cx, cy]],
        labels=[1],  # foreground
    )

    # Visualize point prompt
    point_vis = img_u8.copy()
    cv_radius = max(5, min(w, h) // 80)
    for pt_result in [(cx, cy)]:
        px, py = pt_result
        # Draw crosshair
        point_vis[max(0, py-cv_radius):min(h, py+cv_radius), px-1:px+2] = [0, 255, 0]
        point_vis[py-1:py+2, max(0, px-cv_radius):min(w, px+cv_radius)] = [0, 255, 0]

    # Best mask overlay
    best_idx = point_result['best_idx']
    best_mask = point_result['masks'][best_idx]
    mask_overlay = _overlay_mask(img_u8, best_mask, color=(0, 200, 80))

    # Multi-mask comparison (3 candidates)
    multi_vis = _make_multi_mask_strip(img_u8, point_result['masks'], point_result['scores'])

    steps = [
        {'id': 'original', 'name': '输入图像', 'image': img_u8,
         'explanation': '原始图像。SAM 用 ViT 图像编码器一次性编码整张图（耗时最长的一步），之后可以反复用不同提示做快速分割'},
        {'id': 'point_prompt', 'name': '点提示 (中心点)', 'image': point_vis,
         'explanation': f'绿色十字 = 正样本提示点 (x={cx}, y={cy})。SAM 支持点(正/负)、框、文本三种提示——这是 CV 中"提示工程"的范式'},
        {'id': 'best_mask', 'name': f'最佳分割掩码 (IoU={point_result["scores"][best_idx]:.3f})', 'image': mask_overlay,
         'explanation': f'SAM 从提示点生成 3 个候选掩码，自动选择 IoU 预测最高的一个。绿色区域 = 分割出的物体'},
        {'id': 'multi_masks', 'name': '3 个候选掩码对比', 'image': multi_vis,
         'explanation': f'SAM 输出 3 个不同粒度的候选掩码（整体/部分/子部分），由 Mask Decoder 同时生成，模型自己预测每个掩码的 IoU 分数来选择最佳'},
    ]

    return {
        'steps': steps,
        'metrics': {
            'model': 'SAM (ViT-B)',
            'image_size': f'{w}×{h}',
            'num_candidates': 3,
            'best_iou_score': round(point_result['scores'][best_idx], 3),
            'prompt_type': 'point (click)',
        },
    }


def _overlay_mask(img, mask, color=(0, 200, 80)):
    h, w = img.shape[:2]
    mask_bool = np.asarray(mask) > 128
    if mask_bool.shape[:2] != (h, w):
        from PIL import Image
        mask_bool = np.array(Image.fromarray(mask_bool.astype(np.uint8)*255).resize((w, h), Image.NEAREST)) > 128
    overlay = img.copy().astype(np.float32)
    alpha = 0.45
    for c in range(3):
        overlay[mask_bool, c] = overlay[mask_bool, c] * (1-alpha) + color[c] * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _make_multi_mask_strip(img, masks, scores):
    from PIL import Image
    thumb_h = 150
    thumbs = []
    for i, mask in enumerate(masks):
        overlay = _overlay_mask(img, mask)
        # Resize to thumb
        h, w = overlay.shape[:2]
        scale = thumb_h / h
        new_w = max(1, int(w * scale))
        thumb = np.array(Image.fromarray(overlay).resize((new_w, thumb_h), Image.LANCZOS))
        thumbs.append(thumb)
    total_w = sum(t.shape[1] for t in thumbs) + max(0, len(thumbs)-1)*4
    canvas = np.zeros((thumb_h, total_w, 3), dtype=np.uint8)
    x_off = 0
    for i, t in enumerate(thumbs):
        canvas[:, x_off:x_off+t.shape[1]] = t
        x_off += t.shape[1] + 4
    return canvas


def _fallback_pipeline(img_u8, error_msg):
    steps = [
        {'id': 'original', 'name': '输入图像', 'image': img_u8,
         'explanation': '原始图像'},
        {'id': 'info', 'name': 'SAM 模型状态', 'image': img_u8,
         'explanation': f'SAM 模型未加载。请将 sam_vit_b_01ec64.pth 放到 models/ 目录。错误: {error_msg[:200]}'},
    ]
    return {'steps': steps, 'metrics': {'status': 'model_not_available'}}
