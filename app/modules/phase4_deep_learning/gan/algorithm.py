"""Small real GAN training trace for teaching.

This is not a pretrained image generator. It trains a tiny NumPy GAN on a
synthetic 2D mixture distribution and returns real intermediate results:
real samples, generator samples before/after training, discriminator surface
and loss curves.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def _mm(a, b):
    """Small matrix multiply without BLAS GEMM to avoid mixed-runtime aborts."""
    return np.einsum('ik,kj->ij', np.asarray(a), np.asarray(b), optimize=False)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _real_samples(rng, n):
    centers = np.array([[-1.8, -1.4], [-1.4, 1.5], [1.6, -1.2], [1.4, 1.4]], dtype=np.float64)
    idx = rng.integers(0, len(centers), size=n)
    return centers[idx] + rng.normal(0.0, 0.18, size=(n, 2))


def _init_params(rng):
    return {
        'Wg1': rng.normal(0, 0.45, (2, 16)),
        'bg1': np.zeros(16),
        'Wg2': rng.normal(0, 0.35, (16, 2)),
        'bg2': np.zeros(2),
        'Wd1': rng.normal(0, 0.55, (2, 18)),
        'bd1': np.zeros(18),
        'Wd2': rng.normal(0, 0.45, (18, 1)),
        'bd2': np.zeros(1),
    }


def _G(z, p):
    h = np.tanh(_mm(z, p['Wg1']) + p['bg1'])
    x = _mm(h, p['Wg2']) + p['bg2']
    return x, h


def _D(x, p):
    h = np.tanh(_mm(x, p['Wd1']) + p['bd1'])
    logit = _mm(h, p['Wd2']) + p['bd2']
    prob = _sigmoid(logit)
    return prob, h, logit


def _bce(prob, target):
    eps = 1e-8
    return -np.mean(target * np.log(prob + eps) + (1.0 - target) * np.log(1.0 - prob + eps))


def _train_gan(seed=7, iterations=260, batch=128):
    rng = np.random.default_rng(seed)
    p = _init_params(rng)
    lr_d, lr_g = 0.035, 0.025
    losses = []

    z_probe = rng.normal(0, 1, size=(512, 2))
    initial_fake, _ = _G(z_probe, p)

    for it in range(iterations):
        real = _real_samples(rng, batch)
        z = rng.normal(0, 1, size=(batch, 2))
        fake, _ = _G(z, p)

        # --- discriminator update ---
        x_all = np.vstack([real, fake])
        y_all = np.vstack([np.ones((batch, 1)), np.zeros((batch, 1))])
        pred, h_d, _ = _D(x_all, p)
        dlogit = (pred - y_all) / len(x_all)
        gWd2 = _mm(h_d.T, dlogit)
        gbd2 = dlogit.sum(axis=0)
        dh = _mm(dlogit, p['Wd2'].T) * (1.0 - h_d * h_d)
        gWd1 = _mm(x_all.T, dh)
        gbd1 = dh.sum(axis=0)
        p['Wd2'] -= lr_d * gWd2
        p['bd2'] -= lr_d * gbd2
        p['Wd1'] -= lr_d * gWd1
        p['bd1'] -= lr_d * gbd1

        # --- generator update, discriminator frozen ---
        z = rng.normal(0, 1, size=(batch, 2))
        fake, h_g = _G(z, p)
        pred_fake, h_df, _ = _D(fake, p)
        dlogit_g = (pred_fake - 1.0) / batch
        dhd = _mm(dlogit_g, p['Wd2'].T) * (1.0 - h_df * h_df)
        dfake = _mm(dhd, p['Wd1'].T)
        gWg2 = _mm(h_g.T, dfake)
        gbg2 = dfake.sum(axis=0)
        dhg = _mm(dfake, p['Wg2'].T) * (1.0 - h_g * h_g)
        gWg1 = _mm(z.T, dhg)
        gbg1 = dhg.sum(axis=0)
        p['Wg2'] -= lr_g * gWg2
        p['bg2'] -= lr_g * gbg2
        p['Wg1'] -= lr_g * gWg1
        p['bg1'] -= lr_g * gbg1

        if it % 10 == 0 or it == iterations - 1:
            real_eval = _real_samples(rng, batch)
            fake_eval, _ = _G(rng.normal(0, 1, size=(batch, 2)), p)
            pr, _, _ = _D(real_eval, p)
            pf, _, _ = _D(fake_eval, p)
            losses.append({
                'iter': it,
                'D_loss': float(_bce(pr, 1.0) + _bce(pf, 0.0)),
                'G_loss': float(_bce(pf, 1.0)),
                'D_real': float(pr.mean()),
                'D_fake': float(pf.mean()),
            })

    final_fake, _ = _G(z_probe, p)
    real_vis = _real_samples(rng, 512)
    return p, real_vis, initial_fake, final_fake, losses


def _xy_to_canvas(points, width, height):
    arr = np.asarray(points, dtype=np.float64)
    x = np.clip((arr[:, 0] + 2.8) / 5.6, 0, 1)
    y = np.clip((arr[:, 1] + 2.4) / 4.8, 0, 1)
    return (x * (width - 1)).astype(int), ((1.0 - y) * (height - 1)).astype(int)


def _scatter(real, fake=None, title='GAN samples', width=520, height=360):
    img = Image.new('RGB', (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(img)
    draw.rectangle((42, 24, width - 22, height - 34), outline=(203, 213, 225), width=1)
    draw.line((42, height // 2, width - 22, height // 2), fill=(226, 232, 240))
    draw.line((width // 2, 24, width // 2, height - 34), fill=(226, 232, 240))
    draw.text((18, 6), title, fill=(15, 23, 42))
    rx, ry = _xy_to_canvas(real, width, height)
    for x, y in zip(rx, ry):
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(37, 99, 235))
    if fake is not None:
        fx, fy = _xy_to_canvas(fake, width, height)
        for x, y in zip(fx, fy):
            draw.rectangle((x - 2, y - 2, x + 2, y + 2), fill=(225, 29, 72))
        draw.text((width - 178, 8), 'blue=real  red=generated', fill=(71, 85, 105))
    else:
        draw.text((width - 128, 8), 'blue=real data', fill=(71, 85, 105))
    return np.array(img)


def _loss_chart(losses, width=520, height=260):
    img = Image.new('RGB', (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(img)
    pad_l, pad_t, pad_r, pad_b = 48, 24, 20, 38
    draw.rectangle((pad_l, pad_t, width - pad_r, height - pad_b), outline=(203, 213, 225))
    xs = np.array([d['iter'] for d in losses], dtype=np.float64)
    d_loss = np.array([d['D_loss'] for d in losses], dtype=np.float64)
    g_loss = np.array([d['G_loss'] for d in losses], dtype=np.float64)
    y_max = max(float(d_loss.max()), float(g_loss.max()), 1.0)

    def pts(values):
        x = pad_l + (xs - xs.min()) / max(xs.max() - xs.min(), 1e-8) * (width - pad_l - pad_r)
        y = height - pad_b - values / y_max * (height - pad_t - pad_b)
        return list(zip(x.astype(int), y.astype(int)))

    draw.line(pts(d_loss), fill=(37, 99, 235), width=3)
    draw.line(pts(g_loss), fill=(225, 29, 72), width=3)
    draw.text((16, 6), 'Adversarial losses during real NumPy training', fill=(15, 23, 42))
    draw.text((60, height - 28), 'blue: discriminator loss', fill=(37, 99, 235))
    draw.text((260, height - 28), 'red: generator loss', fill=(225, 29, 72))
    return np.array(img)


def _decision_surface(p, width=420, height=320):
    xs = np.linspace(-2.8, 2.8, width)
    ys = np.linspace(2.4, -2.4, height)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    prob, _, _ = _D(grid, p)
    prob = prob.reshape(height, width)
    r = (prob * 255).astype(np.uint8)
    b = ((1.0 - prob) * 255).astype(np.uint8)
    g = (80 + np.abs(prob - 0.5) * 120).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def build_pipeline(image_path=None, **kwargs):
    iterations = int(kwargs.get('iterations', 260))
    iterations = max(80, min(iterations, 400))
    p, real, fake0, fake1, losses = _train_gan(iterations=iterations)
    final = losses[-1]
    return {
        'steps': [
            {
                'id': 'real_distribution',
                'name': '真实数据分布',
                'image': _scatter(real, None, 'Real target distribution'),
                'explanation': '这里用四个二维高斯团作为“真实图像数据”的可视化替身。GAN 的目标不是记住样本，而是让生成分布接近真实分布。',
                'formula': 'x ~ p_data(x)',
                'data': {'sample_count': int(len(real)), 'distribution': '4 Gaussian modes'},
            },
            {
                'id': 'initial_generator',
                'name': '训练前生成器',
                'image': _scatter(real, fake0, 'Before training: G(z) does not match data'),
                'explanation': '生成器从随机噪声 z 出发，训练前的 G(z) 分布和真实数据明显不一致。红点是真实前向计算得到的生成样本。',
                'formula': 'z ~ N(0,I), x_fake = G_theta(z)',
            },
            {
                'id': 'adversarial_training',
                'name': '对抗训练损失',
                'image': _loss_chart(losses),
                'explanation': '判别器学习区分真样本和假样本；生成器反向利用判别器梯度，让假样本更像真样本。曲线来自本地 NumPy 反向传播训练。',
                'formula': 'min_G max_D E[log D(x)] + E[log(1-D(G(z)))]',
                'data': {'losses': losses},
            },
            {
                'id': 'discriminator_surface',
                'name': '判别器决策面',
                'image': _decision_surface(p),
                'explanation': '红色区域更像判别器认为的真实区域，蓝色区域更像假区域。生成器的梯度来自这张“地形图”。',
                'formula': 'D_phi(x)=sigmoid(f_phi(x))',
            },
            {
                'id': 'trained_generator',
                'name': '训练后生成结果',
                'image': _scatter(real, fake1, 'After training: generated samples move toward real modes'),
                'explanation': '训练后红点被推向真实数据的几个主要模式。这是一个小型真实 GAN 训练结果，不是预训练图像生成模型。',
                'formula': 'theta <- theta - eta * grad_theta L_G',
                'data': {'D_real': round(final['D_real'], 4), 'D_fake': round(final['D_fake'], 4)},
            },
        ],
        'metrics': {
            'status': 'local_mechanism',
            'backend': 'NumPy',
            'algorithm': 'tiny GAN with real backpropagation',
            'real_model': False,
            'iterations': iterations,
            'D_loss_final': round(final['D_loss'], 4),
            'G_loss_final': round(final['G_loss'], 4),
            'note': '本页展示真实本地 GAN 训练机制，不是大型预训练图像生成器。',
        },
    }
