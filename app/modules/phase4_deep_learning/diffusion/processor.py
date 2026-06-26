"""Diffusion demo — lightweight NumPy simulation + optional real SD."""
import numpy as np
from app.utils.image_utils import load_image_u8, ensure_gray


def _noise_step(img, t, total_steps=50):
    """Add Gaussian noise at level t/total_steps (forward process)."""
    alpha = 1.0 - (t / total_steps) * 0.9
    noise = np.random.randn(*img.shape).astype(np.float32) * 30
    noisy = (img.astype(np.float32) * alpha + noise * (1 - alpha)).clip(0, 255)
    return noisy.astype(np.uint8)


def _denoise_heuristic(noisy, t, total_steps=50):
    """Simple denoising: bilateral-like smoothing (reverse process demo)."""
    from scipy.ndimage import uniform_filter
    strength = int(1 + t * 4 / total_steps)
    denoised = uniform_filter(noisy.astype(np.float32), size=max(3, strength * 2 + 1))
    return denoised.clip(0, 255).astype(np.uint8)


def build_pipeline(image_path=None, prompt='a cat', num_steps=20, **kwargs):
    steps = int(kwargs.get('num_steps', num_steps))
    steps = max(4, min(steps, 30))

    # Try real SD first (fast path if model already cached)
    real = kwargs.get('real', '0')
    if str(real).strip().lower() in ('1', 'true', 'yes'):
        try:
            from app.modules.phase5_frontier.stable_diffusion.algorithm import generate_image
            result = generate_image(prompt=prompt, num_inference_steps=min(steps, 20),
                                    guidance_scale=7.5, height=256, width=256)
            final = np.asarray(result['image'], dtype=np.uint8)
            return {'steps': [
                {'id': 'prompt', 'name': f'prompt: {prompt}', 'image': final,
                 'explanation': f'Stable Diffusion 真实生成，{steps}步推理。'},
                {'id': 'result', 'name': '真实生成结果', 'image': final,
                 'explanation': 'runwayml/stable-diffusion-v1-5 diffusers pipeline 输出。'},
            ], 'metrics': {'model': 'sd-v1-5', 'backend': 'diffusers', 'prompt': prompt, 'steps': steps}}
        except Exception as e:
            pass  # Fall through to NumPy demo

    # NumPy demo: forward (noise) + reverse (denoise) visual story
    img_u8 = load_image_u8(image_path, mode='rgb', max_side=256) if image_path else (
        np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))

    img_gray = ensure_gray(img_u8)
    if img_gray.ndim == 2:
        img_gray = np.stack([img_gray] * 3, axis=-1)

    mid = steps // 2
    noisy = _noise_step(img_gray, mid, steps)
    noisy_full = _noise_step(img_gray, steps - 1, steps)
    denoised = _denoise_heuristic(noisy_full, 3, steps)

    return {
        'steps': [
            {'id': 'original', 'name': '原图', 'image': img_u8,
             'explanation': '输入图像。扩散模型的核心思想：学会从噪声中恢复图像。'},
            {'id': 'noise_mid', 'name': f'加噪 {mid}/{steps} 步', 'image': noisy,
             'explanation': f'前向过程：逐步添加高斯噪声。到第{mid}步时，图像已明显模糊。'},
            {'id': 'noise_full', 'name': f'加噪 {steps}/{steps} 步（近纯噪声）', 'image': noisy_full,
             'explanation': f'前向过程终点：图像几乎变成纯高斯噪声。扩散模型要学的就是从这一步"逆回去"。'},
            {'id': 'denoised', 'name': '去噪还原', 'image': denoised,
             'explanation': '反向过程：UNet预测每步的噪声分量并减去。这就是扩散模型"生成"的本质——从噪声一步步去噪。'},
        ],
        'metrics': {
            'status': 'numpy_demo',
            'backend': 'NumPy + SciPy',
            'model': 'diffusion_concept_demo',
            'steps': steps,
            'note': 'NumPy模拟前向加噪+反向去噪概念。真实SD生成请用Phase5的Stable Diffusion模块。',
        }
    }
