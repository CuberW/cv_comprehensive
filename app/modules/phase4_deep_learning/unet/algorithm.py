"""U-Net-style encoder/decoder segmentation mechanism.

The project does not ship a trained U-Net checkpoint.  This implementation is
a deterministic local mechanism: it runs real image operations that mirror the
encoder, bottleneck, decoder and skip-fusion data flow, and it is explicitly
reported as a local mechanism rather than pretrained semantic segmentation.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from app.modules.offline_teaching import _load_or_fixture
from app.utils.image_utils import ensure_gray, load_image_u8


def build_pipeline(image_path=None, threshold='otsu', **kwargs):
    img = _load_image(image_path)
    gray = ensure_gray(img).astype(np.float32) / 255.0
    h, w = gray.shape

    # Encoder: shallow detail, then progressively lower resolution context.
    luma_smooth = _gaussian_blur(gray, sigma=1.0)
    gy, gx = np.gradient(luma_smooth)
    edge = _norm01(np.sqrt(gx * gx + gy * gy))
    local_contrast = _norm01(np.abs(gray - _gaussian_blur(gray, sigma=5.0)))
    enc1 = _norm01(0.58 * edge + 0.42 * local_contrast)

    enc2 = _norm01(_downsample_mean(enc1, 2))
    enc2_context = _gaussian_blur(enc2, sigma=1.4)
    enc3 = _norm01(_downsample_mean(enc2_context, 2))
    bottleneck = _norm01(_gaussian_blur(enc3, sigma=1.8))

    # Decoder: upsample context and fuse same-scale encoder details.
    dec2_up = _resize_float(bottleneck, enc2.shape)
    dec2 = _norm01(0.62 * dec2_up + 0.38 * enc2)
    dec1_up = _resize_float(dec2, enc1.shape)
    no_skip = _norm01(dec1_up)
    skip_fusion = _norm01(0.56 * dec1_up + 0.30 * enc1 + 0.14 * local_contrast)

    probability = _sigmoid_like(skip_fusion)
    tau = _otsu_threshold(probability) if str(threshold).lower() == 'otsu' else float(threshold)
    if tau > 1:
        tau /= 255.0
    tau = float(np.clip(tau, 0.05, 0.95))
    mask = probability >= tau
    mask = _clean_mask(mask)
    overlay = _overlay_mask(img, mask)
    no_skip_mask = no_skip >= _otsu_threshold(no_skip)
    comparison = _side_by_side([
        ('无跳跃连接', _overlay_mask(img, no_skip_mask, color=(245, 158, 11))),
        ('跳跃连接融合', overlay),
    ])
    arch = _architecture_card(h, w, enc1.shape, enc2.shape, bottleneck.shape)

    return {
        'module_id': 'unet',
        'steps': [
            {
                'id': 'input',
                'name': '输入图像',
                'image': img,
                'formula': 'I in R^{H x W x 3}',
                'explanation': 'U-Net 处理的是整张图像到整张 mask 的映射。这里用上传图像或内置样例作为真实输入。',
                'data': {'height': int(h), 'width': int(w), 'channels': 3},
            },
            {
                'id': 'architecture',
                'name': 'U 形编解码结构',
                'image': arch,
                'formula': 'D_l = concat(up(D_{l+1}), E_l)',
                'explanation': '左侧编码器不断压缩空间尺寸以获得上下文，右侧解码器逐层恢复尺寸；同尺度跳跃连接把浅层边界细节送回解码器。',
                'data': {
                    'encoder_l1': list(map(int, enc1.shape)),
                    'encoder_l2': list(map(int, enc2.shape)),
                    'bottleneck': list(map(int, bottleneck.shape)),
                },
            },
            {
                'id': 'encoder_detail',
                'name': '编码器浅层：边界与局部纹理',
                'image': _gray_rgb(enc1),
                'formula': 'E_1 = concat(|grad(G*Y)|, |Y-G_rho*Y|)',
                'explanation': '浅层特征保留边缘、纹理和局部对比度。真实 U-Net 会用卷积学习这些特征；这里用可解释的梯度和局部对比计算得到。',
                'data': {
                    'edge_mean': round(float(edge.mean()), 4),
                    'contrast_mean': round(float(local_contrast.mean()), 4),
                },
            },
            {
                'id': 'bottleneck',
                'name': '瓶颈层：低分辨率上下文',
                'image': _resize_nearest_rgb(_gray_rgb(bottleneck), (h, w)),
                'formula': 'B = down(down(E_1))',
                'explanation': '连续下采样让每个位置看到更大范围的图像结构，语义更强但空间细节更少。',
                'data': {'downsample_ratio': 4, 'bottleneck_shape': list(map(int, bottleneck.shape))},
            },
            {
                'id': 'decoder_no_skip',
                'name': '仅上采样解码：边界容易变粗',
                'image': _gray_rgb(no_skip),
                'formula': 'D_1 = up(up(B))',
                'explanation': '如果只把瓶颈层放大回原图，区域结构能回来，但边界会更粗、更模糊。这正是 U-Net 需要跳跃连接的原因。',
                'data': {'mean_response': round(float(no_skip.mean()), 4)},
            },
            {
                'id': 'skip_fusion',
                'name': '跳跃连接融合细节',
                'image': _gray_rgb(skip_fusion),
                'formula': 'P = sigmoid(w_d D_1 + w_e E_1 + w_c C)',
                'explanation': '解码器的上下文响应与编码器浅层细节融合，既知道大致区域，也尽量贴住局部边界。',
                'data': {'decoder_weight': 0.56, 'encoder_weight': 0.30, 'contrast_weight': 0.14},
            },
            {
                'id': 'probability',
                'name': '前景概率图',
                'image': _heatmap(probability),
                'formula': 'p(x,y)=sigmoid(P(x,y))',
                'explanation': '概率图越亮的位置越可能被分成前景。它是阈值化之前的连续中间结果。',
                'data': {
                    'threshold': round(tau, 4),
                    'mean_probability': round(float(probability.mean()), 4),
                },
            },
            {
                'id': 'mask',
                'name': '二值 mask 与结果叠加',
                'image': overlay,
                'formula': 'M(x,y)=1[p(x,y)>=tau]',
                'explanation': '按阈值把概率图变成二值 mask，再叠加到原图上。该结果来自本地机制计算，不是训练好的医学或通用 U-Net 权重。',
                'data': {
                    'foreground_ratio': round(float(mask.mean()), 4),
                    'foreground_pixels': int(mask.sum()),
                },
            },
            {
                'id': 'skip_comparison',
                'name': '跳跃连接前后对比',
                'image': comparison,
                'formula': 'mask_skip vs mask_no_skip',
                'explanation': '左边只用瓶颈上下文，右边加入跳跃连接。对比展示 U-Net 为什么能在恢复分辨率时保留边界。',
                'data': {
                    'no_skip_foreground_ratio': round(float(no_skip_mask.mean()), 4),
                    'skip_foreground_ratio': round(float(mask.mean()), 4),
                },
            },
        ],
        'metrics': {
            'status': 'local_mechanism',
            'backend': 'NumPy/Pillow',
            'algorithm': 'U-Net-style encoder/decoder with skip fusion',
            'real_pretrained_unet': False,
            'threshold': round(tau, 4),
            'foreground_ratio': round(float(mask.mean()), 4),
            'note': '本页展示 U-Net 数据流和跳跃连接机制；不冒充训练好的 U-Net 权重。',
        },
    }


def _load_image(image_path):
    if image_path:
        return load_image_u8(image_path, mode='rgb', max_side=512)
    return _load_or_fixture(image_path=image_path)


def _gaussian_blur(arr, sigma=1.0):
    arr = np.asarray(arr, dtype=np.float32)
    radius = max(1, int(math.ceil(3 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(xs * xs) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    out = _convolve_axis(arr, kernel, axis=0)
    out = _convolve_axis(out, kernel, axis=1)
    return out.astype(np.float32)


def _convolve_axis(arr, kernel, axis):
    pad = len(kernel) // 2
    padded = np.pad(arr, [(pad, pad) if i == axis else (0, 0) for i in range(arr.ndim)], mode='edge')
    out = np.zeros_like(arr, dtype=np.float32)
    for k, weight in enumerate(kernel):
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(k, k + arr.shape[axis])
        out += float(weight) * padded[tuple(sl)]
    return out


def _downsample_mean(arr, factor):
    h, w = arr.shape
    h2 = max(1, h // factor)
    w2 = max(1, w // factor)
    crop = arr[:h2 * factor, :w2 * factor]
    if crop.size == 0:
        return arr.copy()
    return crop.reshape(h2, factor, w2, factor).mean(axis=(1, 3))


def _resize_float(arr, shape_hw):
    h, w = shape_hw
    img = Image.fromarray((_norm01(arr) * 255).astype(np.uint8), mode='L').resize((w, h), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def _resize_nearest_rgb(arr, shape_hw):
    h, w = shape_hw
    return np.asarray(Image.fromarray(arr).resize((w, h), Image.NEAREST))


def _norm01(x):
    x = np.asarray(x, dtype=np.float32)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


def _sigmoid_like(x):
    z = (_norm01(x) - 0.5) * 7.0
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _otsu_threshold(values):
    vals = np.clip(values, 0, 1)
    hist, _ = np.histogram(vals, bins=256, range=(0, 1))
    total = vals.size
    prob = hist.astype(np.float64) / max(1, total)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    valid = denom > 1e-12
    score = np.zeros(256, dtype=np.float64)
    score[valid] = ((mu_t * omega[valid] - mu[valid]) ** 2) / denom[valid]
    return int(np.argmax(score)) / 255.0


def _clean_mask(mask):
    mask = np.asarray(mask, dtype=bool)
    count = np.zeros(mask.shape, dtype=np.int16)
    padded = np.pad(mask, 1, mode='edge')
    for dy in range(3):
        for dx in range(3):
            count += padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return count >= 4


def _gray_rgb(arr):
    g = (_norm01(arr) * 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def _heatmap(values):
    v = np.clip(values, 0, 1)
    out = np.zeros((*v.shape, 3), dtype=np.float32)
    out[..., 0] = 239 * v + 15 * (1 - v)
    out[..., 1] = 68 * v + 23 * (1 - v)
    out[..., 2] = 68 * v + 42 * (1 - v)
    return np.clip(out, 0, 255).astype(np.uint8)


def _overlay_mask(img, mask, color=(34, 197, 94)):
    out = img.astype(np.float32).copy()
    col = np.array(color, dtype=np.float32)
    out[mask] = out[mask] * 0.42 + col * 0.58
    edge = _mask_edges(mask)
    out[edge] = np.array([250, 204, 21], dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _mask_edges(mask):
    edge = np.zeros(mask.shape, dtype=bool)
    edge[1:, :] |= mask[1:, :] != mask[:-1, :]
    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    return edge


def _side_by_side(items):
    thumbs = []
    labels = []
    for label, image in items:
        im = Image.fromarray(image).resize((300, 220), Image.BILINEAR)
        thumbs.append(im)
        labels.append(label)
    canvas = Image.new('RGB', (300 * len(thumbs), 252), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    for i, im in enumerate(thumbs):
        x = i * 300
        canvas.paste(im, (x, 0))
        draw.rectangle((x, 220, x + 300, 252), fill=(15, 23, 42))
        draw.text((x + 12, 230), labels[i], fill=(226, 232, 240))
    return np.array(canvas)


def _architecture_card(h, w, enc1_shape, enc2_shape, bottleneck_shape):
    width, height = 760, 320
    canvas = Image.new('RGB', (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    blocks = [
        ('Encoder L1', f'{enc1_shape[0]}x{enc1_shape[1]}', 36, 58, (59, 130, 246)),
        ('Encoder L2', f'{enc2_shape[0]}x{enc2_shape[1]}', 180, 94, (37, 99, 235)),
        ('Bottleneck', f'{bottleneck_shape[0]}x{bottleneck_shape[1]}', 330, 132, (124, 58, 237)),
        ('Decoder L2', f'{enc2_shape[0]}x{enc2_shape[1]}', 480, 94, (20, 184, 166)),
        ('Output mask', f'{h}x{w}', 624, 58, (34, 197, 94)),
    ]
    for title, dim, x, y, color in blocks:
        draw.rounded_rectangle((x, y, x + 104, y + 76), radius=8, fill=color)
        draw.text((x + 10, y + 20), title, fill=(255, 255, 255))
        draw.text((x + 10, y + 46), dim, fill=(226, 232, 240))
    for x0, y0, x1, y1 in [(140, 96, 180, 120), (284, 132, 330, 158), (434, 158, 480, 132), (584, 120, 624, 96)]:
        draw.line((x0, y0, x1, y1), fill=(71, 85, 105), width=3)
        draw.polygon([(x1, y1), (x1 - 10, y1 - 5), (x1 - 10, y1 + 5)], fill=(71, 85, 105))
    draw.arc((90, 28, 676, 252), 195, 345, fill=(245, 158, 11), width=3)
    draw.text((266, 26), 'skip connection: shallow details return to decoder', fill=(146, 64, 14))
    draw.text((28, 278), '本地机制实现：真实计算 U-Net 数据流，不冒充训练好的 U-Net 权重。', fill=(51, 65, 85))
    return np.array(canvas)
