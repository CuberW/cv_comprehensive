"""Real diffusion pipeline using diffusers Stable Diffusion."""
import numpy as np


def build_pipeline(image_path=None, prompt='a cat sitting on a chair', num_steps=20, guidance=7.5, **kwargs):
    p = kwargs.get('prompt', prompt)
    steps = int(kwargs.get('num_steps', num_steps))
    cfg = float(kwargs.get('guidance', guidance))

    try:
        from app.modules.phase5_frontier.stable_diffusion.algorithm import generate_image
        result = generate_image(prompt=p, num_inference_steps=min(steps, 30), guidance_scale=cfg, height=384, width=384)
        final = np.asarray(result['image'], dtype=np.uint8)
    except Exception as e:
        return {
            'error': f'Stable Diffusion 模型未就绪: {e}',
            'steps': [],
            'metrics': {'status': 'model_not_available'},
        }

    return {
        'steps': [
            {
                'id': 'prompt',
                'name': '文本提示',
                'image': final,
                'explanation': f'prompt = {p}',
            },
            {
                'id': 'result',
                'name': '真实生成结果',
                'image': final,
                'explanation': '来自 diffusers StableDiffusionPipeline 的真实输出。',
            },
        ],
        'metrics': {
            'model': 'runwayml/stable-diffusion-v1-5',
            'backend': 'diffusers',
            'prompt': p,
            'inference_steps': steps,
            'guidance_scale': cfg,
        },
    }
