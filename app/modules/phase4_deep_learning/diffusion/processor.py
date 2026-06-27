"""DDPM teaching pipeline with real NumPy forward/reverse equations.

This module intentionally does not pretend to be Stable Diffusion. It computes
the DDPM noising equation on the uploaded image, visualizes the exact noise
term, and reconstructs with the oracle epsilon so every displayed result comes
from backend math instead of frontend illustration.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from app.modules.offline_teaching import _load_or_fixture


def _schedule(num_steps):
    betas = np.linspace(0.0005, 0.045, num_steps, dtype=np.float64)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    return betas, alphas, alpha_bar


def _to_float(rgb):
    return np.asarray(rgb, dtype=np.float64) / 255.0


def _to_u8(x):
    return np.round(np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def _noise_to_image(eps):
    e = np.asarray(eps, dtype=np.float64)
    e = (e - e.min()) / max(float(e.max() - e.min()), 1e-8)
    return _to_u8(e)


def _xt(x0, eps, alpha_bar_t):
    return np.sqrt(alpha_bar_t) * x0 + np.sqrt(1.0 - alpha_bar_t) * eps


def _oracle_reconstruct(xt, eps, alpha_bar_t):
    return (xt - np.sqrt(1.0 - alpha_bar_t) * eps) / max(np.sqrt(alpha_bar_t), 1e-8)


def _strip(images, labels, thumb_h=132):
    thumbs = []
    for img in images:
        pil = Image.fromarray(img)
        scale = thumb_h / max(1, pil.height)
        thumbs.append(pil.resize((max(1, int(pil.width * scale)), thumb_h), Image.BILINEAR))
    label_h = 26
    total_w = sum(t.width for t in thumbs) + 8 * (len(thumbs) - 1)
    canvas = Image.new('RGB', (total_w, thumb_h + label_h), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for img, label in zip(thumbs, labels):
        canvas.paste(img, (x, label_h))
        draw.text((x + 4, 5), label, fill=(15, 23, 42))
        x += img.width + 8
    return np.array(canvas)


def _curve_chart(alpha_bar, selected, width=540, height=260):
    img = Image.new('RGB', (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = 44, 22, 18, 36
    draw.rectangle((left, top, width - right, height - bottom), outline=(203, 213, 225))
    xs = np.linspace(left, width - right, len(alpha_bar))
    ys = height - bottom - alpha_bar * (height - top - bottom)
    pts = list(zip(xs.astype(int), ys.astype(int)))
    draw.line(pts, fill=(37, 99, 235), width=3)
    for t in selected:
        x = int(xs[t])
        y = int(ys[t])
        draw.line((x, top, x, height - bottom), fill=(225, 29, 72), width=1)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(225, 29, 72))
        draw.text((x + 4, y - 14), f't={t + 1}', fill=(100, 116, 139))
    draw.text((16, 4), 'Noise schedule: alpha_bar keeps shrinking, noise weight grows', fill=(15, 23, 42))
    draw.text((52, height - 26), 'time step', fill=(71, 85, 105))
    draw.text((6, 110), 'alpha_bar', fill=(71, 85, 105))
    return np.array(img)


def build_pipeline(image_path=None, num_steps=40, **kwargs):
    steps_count = max(8, min(int(kwargs.get('num_steps', num_steps)), 80))
    rgb = _load_or_fixture(image_path=image_path)
    if max(rgb.shape[:2]) > 256:
        pil = Image.fromarray(rgb)
        pil.thumbnail((256, 256), Image.BILINEAR)
        rgb = np.array(pil)
    x0 = _to_float(rgb)

    rng = np.random.default_rng(123)
    eps = rng.normal(0.0, 1.0, size=x0.shape)
    _, _, alpha_bar = _schedule(steps_count)

    t_early = max(0, steps_count // 5 - 1)
    t_mid = max(0, steps_count // 2 - 1)
    t_late = steps_count - 1
    xt_early = _xt(x0, eps, alpha_bar[t_early])
    xt_mid = _xt(x0, eps, alpha_bar[t_mid])
    xt_late = _xt(x0, eps, alpha_bar[t_late])
    recon = _oracle_reconstruct(xt_late, eps, alpha_bar[t_late])
    err = np.abs(recon - x0)

    timeline = _strip(
        [_to_u8(x0), _to_u8(xt_early), _to_u8(xt_mid), _to_u8(xt_late), _to_u8(recon)],
        ['x0', f't={t_early + 1}', f't={t_mid + 1}', f't={t_late + 1}', 'x0_hat'],
    )
    mse = float(np.mean((recon - x0) ** 2))

    return {
        'steps': [
            {
                'id': 'input',
                'name': '输入图像 x0',
                'image': _to_u8(x0),
                'explanation': '扩散模型把干净图像看作 x0，后续每个时间步都会按噪声日程混入更强的高斯噪声。',
                'formula': 'x0 in [0,1]^{H x W x 3}',
            },
            {
                'id': 'schedule',
                'name': '噪声日程 alpha_bar',
                'image': _curve_chart(alpha_bar, [t_early, t_mid, t_late]),
                'explanation': 'alpha_bar_t 越小，原图权重越低，噪声权重越高。图中的红线对应本页展示的几个时间步。',
                'formula': 'alpha_bar_t = product_{s=1..t}(1 - beta_s)',
                'data': {'alpha_bar': [round(float(v), 6) for v in alpha_bar.tolist()]},
            },
            {
                'id': 'noise',
                'name': '采样到的高斯噪声 epsilon',
                'image': _noise_to_image(eps),
                'explanation': '这是后端真实采样的标准高斯噪声。DDPM 的训练目标就是让网络学会预测每一步里的这部分噪声。',
                'formula': 'epsilon ~ N(0, I)',
            },
            {
                'id': 'forward_early',
                'name': f'前向加噪 t={t_early + 1}',
                'image': _to_u8(xt_early),
                'explanation': '早期时间步仍保留大部分原图结构，只叠加少量噪声。',
                'formula': 'x_t = sqrt(alpha_bar_t) x0 + sqrt(1-alpha_bar_t) epsilon',
            },
            {
                'id': 'forward_mid',
                'name': f'前向加噪 t={t_mid + 1}',
                'image': _to_u8(xt_mid),
                'explanation': '中间时间步里噪声已经明显增强，图像语义开始被破坏。',
                'formula': 'q(x_t | x0) = N(sqrt(alpha_bar_t)x0, (1-alpha_bar_t)I)',
            },
            {
                'id': 'forward_late',
                'name': f'接近纯噪声 t={t_late + 1}',
                'image': _to_u8(xt_late),
                'explanation': '后期时间步中原图权重很低，模型在生成时需要从类似这种噪声状态一步步反推。',
                'formula': 'noise_weight = sqrt(1-alpha_bar_t)',
            },
            {
                'id': 'reverse_oracle',
                'name': '反向还原示例',
                'image': _to_u8(recon),
                'explanation': '这里用已知 epsilon 做 oracle 还原，展示 DDPM 反向公式的含义。真实生成模型会用神经网络 epsilon_theta 预测 epsilon。',
                'formula': 'x0_hat = (x_t - sqrt(1-alpha_bar_t) epsilon_theta) / sqrt(alpha_bar_t)',
                'data': {'mse_to_input': mse},
            },
            {
                'id': 'timeline',
                'name': '从清晰到噪声再回到图像',
                'image': timeline,
                'explanation': '这条时间轴把前向加噪和反向重建放在一起。它是真实 NumPy 计算结果，不是前端静态示意。',
                'formula': 'forward q(x_t|x0), reverse p_theta(x_{t-1}|x_t)',
            },
            {
                'id': 'error',
                'name': '重建误差图',
                'image': _to_u8(err / max(float(err.max()), 1e-8)),
                'explanation': '误差图用于说明：如果噪声预测准确，反向公式可以恢复接近原图的结果；真实扩散模型的难点在于学习 epsilon_theta。',
                'formula': 'MSE = mean((x0_hat - x0)^2)',
            },
        ],
        'metrics': {
            'status': 'local_mechanism',
            'backend': 'NumPy DDPM equations',
            'real_model': False,
            'steps': steps_count,
            'alpha_bar_final': round(float(alpha_bar[-1]), 6),
            'reconstruction_mse': round(mse, 8),
            'note': '本页展示真实 DDPM 数学机制，不是 Stable Diffusion 预训练采样。',
        },
    }
