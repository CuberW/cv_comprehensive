"""DCGAN forward pass — real computation."""
import numpy as np
import io, base64
from PIL import Image
from app.utils.image_utils import load_image_u8

def _b64(arr):
    b = io.BytesIO(); Image.fromarray(arr).save(b, 'PNG')
    return base64.b64encode(b.getvalue()).decode()

def build_pipeline(image_path=None, **kwargs):
    rng = np.random.default_rng(42)
    z_dim = 100

    # Generate from noise via transposed conv layers (real architecture)
    z = rng.normal(0, 1, z_dim).astype(np.float64)

    # Layer 1: dense + reshape to 4x4x64
    W1 = rng.normal(0, 0.02, (z_dim, 4*4*64)).astype(np.float64)
    h = np.maximum(0, (z @ W1).reshape(4, 4, 64))

    # Layer 2: 4x4→8x8
    h2 = np.zeros((8, 8, 32), dtype=np.float64)
    for c in range(32):
        k = rng.normal(0, 0.02, (3,3)).astype(np.float64)
        up = np.zeros((8, 8), dtype=np.float64)
        up[::2, ::2] = h[:,:,c%64]
        p = np.pad(up, 1, mode='constant')
        for y in range(8):
            for x in range(8):
                h2[y,x,c] = np.sum(p[y:y+3, x:x+3] * k)
    h2 = np.maximum(0, h2)

    # Layer 3: 8x8→16x16
    h3 = np.zeros((16, 16, 16), dtype=np.float64)
    for c in range(16):
        k = rng.normal(0, 0.02, (3,3)).astype(np.float64)
        up = np.zeros((16, 16), dtype=np.float64)
        up[::2, ::2] = h2[:,:,c%32]
        p = np.pad(up, 1, mode='constant')
        for y in range(16):
            for x in range(16):
                h3[y,x,c] = np.sum(p[y:y+3, x:x+3] * k)
    h3 = np.maximum(0, h3)

    # Layer 4: 16x16→32x32→RGB
    out = np.zeros((32, 32, 3), dtype=np.float64)
    for c in range(3):
        k = rng.normal(0, 0.02, (3,3)).astype(np.float64)
        up = np.zeros((32, 32), dtype=np.float64)
        up[::2, ::2] = h3[:,:,c%16]
        p = np.pad(up, 1, mode='constant')
        for y in range(32):
            for x in range(32):
                out[y,x,c] = np.sum(p[y:y+3, x:x+3] * k)
    generated = np.clip((np.tanh(out) + 1) / 2 * 255, 0, 255).astype(np.uint8)

    # Discriminator evaluation on generated image
    gray_gen = (0.299*generated[:,:,0] + 0.587*generated[:,:,1] + 0.114*generated[:,:,2]).astype(np.float64)
    pool_gen = gray_gen[::4, ::4].ravel()
    Wd = rng.normal(0, 0.02, (min(64, len(pool_gen)), 1)).astype(np.float64)
    fake_score = float(1.0 / (1.0 + np.exp(-float(pool_gen[:len(Wd)] @ Wd))))

    # Noise visualization
    z_vis = np.zeros((z_dim, 20, 3), dtype=np.uint8) + 30
    z_norm = ((z - z.min()) / max(z.max()-z.min(), 1e-8) * 255).astype(np.uint8)
    for i in range(z_dim):
        z_vis[i, :, 0] = z_norm[i]

    return {'steps': [
        {'id': 'noise', 'name': f'输入噪声 ({z_dim}维)', 'image': _b64(z_vis),
         'explanation': f'从标准正态分布采样的{z_dim}维随机向量。真实DCGAN生成器的起点。'},
        {'id': 'layer1', 'name': 'Dense→4×4×64 (ReLU)', 'image': _b64(np.clip((h[:,:,0]-h[:,:,0].min())/max(h[:,:,0].max()-h[:,:,0].min(),1e-8)*255,0,255).astype(np.uint8)),
         'explanation': '全连接层将噪声映射为4×4×64的特征图。'},
        {'id': 'layer3', 'name': '8×8→16×16 (ReLU)', 'image': _b64(np.clip((h3[:,:,0]-h3[:,:,0].min())/max(h3[:,:,0].max()-h3[:,:,0].min(),1e-8)*255,0,255).astype(np.uint8)),
         'explanation': '转置卷积上采样，特征图分辨率逐层翻倍。'},
        {'id': 'generated', 'name': '生成结果 (32×32 RGB)', 'image': _b64(generated),
         'explanation': f'经Tanh激活后的RGB输出。判别器评分={fake_score:.3f}（真实DCGAN有训练权重，此处展示架构前向传播）。'},
    ], 'metrics': {
        'status': 'numpy_algorithm', 'backend': 'NumPy DCGAN',
        'z_dim': z_dim, 'output': '32x32',
        'generator_layers': 4, 'fake_score': round(fake_score, 4),
    }}
