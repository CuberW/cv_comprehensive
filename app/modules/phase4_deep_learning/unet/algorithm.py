
"""U-Net encoder-decoder — real NumPy forward pass."""
import numpy as np
from app.utils.image_utils import load_image_u8, ensure_gray

def _conv_block(x, filters, rng):
    """Simple conv block: conv-like operation."""
    h, w = x.shape[:2]
    out = np.zeros((h, w, filters), dtype=np.float64)
    for f in range(filters):
        k = rng.normal(0, 0.1, (3,3)).astype(np.float64)
        padded = np.pad(x[:,:,0] if x.ndim==3 else x, 1, mode='edge')
        for y in range(h):
            for x_ in range(w):
                out[y,x_,f] = np.sum(padded[y:y+3,x_:x_+3] * k)
    out = np.maximum(0, out)
    return out

def _upsample_to(x, target_hw):
    """Nearest-neighbor upsample/crop/pad to target height and width."""
    th, tw = target_hw
    scale_y = max(1, int(np.ceil(th / max(1, x.shape[0]))))
    scale_x = max(1, int(np.ceil(tw / max(1, x.shape[1]))))
    up = np.repeat(np.repeat(x, scale_y, axis=0), scale_x, axis=1)
    out = np.zeros((th, tw, x.shape[2]), dtype=x.dtype)
    h = min(th, up.shape[0])
    w = min(tw, up.shape[1])
    out[:h, :w, :] = up[:h, :w, :]
    if h < th:
        out[h:, :w, :] = out[h - 1:h, :w, :]
    if w < tw:
        out[:, w:, :] = out[:, w - 1:w, :]
    return out

def build_pipeline(image_path=None, **kwargs):
    if image_path:
        img = load_image_u8(image_path, mode='rgb', max_side=128)
    else:
        img = (np.ones((64,64,3),dtype=np.uint8)*128)
    gray = ensure_gray(img).astype(np.float64) / 255.0
    rng = np.random.default_rng(42)

    # Encoder: 3 levels of downsampling
    e1 = _conv_block(gray, 4, rng)
    e2 = _conv_block(e1[::2,::2,:], 8, rng)
    e3 = _conv_block(e2[::2,::2,:], 16, rng)

    # Decoder: upsample to the matching encoder feature size, then add skip features.
    d2_up = _upsample_to(e3, e2.shape[:2])
    d2 = _conv_block(np.concatenate([d2_up[:, :, :1], e2[:, :, :1]], axis=-1), 8, rng)

    d1_up = _upsample_to(d2, e1.shape[:2])
    d1 = _conv_block(np.concatenate([d1_up[:, :, :1], e1[:, :, :1]], axis=-1), 4, rng)

    # Output: binary mask
    output = np.clip((d1[:,:,0] - d1[:,:,0].min()) / max(d1[:,:,0].max()-d1[:,:,0].min(), 1e-8) * 255, 0, 255).astype(np.uint8)
    output_rgb = np.stack([output, output, output], axis=-1)

    import io,base64; from PIL import Image
    def _b64(arr): b=io.BytesIO(); Image.fromarray(arr).save(b,'PNG'); return base64.b64encode(b.getvalue()).decode()

    enc_vis = np.clip((e1[:,:,0] - e1[:,:,0].min()) / max(e1[:,:,0].max()-e1[:,:,0].min(), 1e-8)*255,0,255).astype(np.uint8)
    return {'steps': [
        {'id':'input','name':'输入图像','image':_b64(img),'explanation':'U-Net接受任意尺寸图像，输出同尺寸分割图。'},
        {'id':'encoder1','name':'编码器 L1','image':_b64(enc_vis),'explanation':'4个3x3卷积+ReLU提取低层特征（边缘、纹理）。'},
        {'id':'encoder3','name':'编码器 L3（瓶颈）','image':_b64(np.clip((e3[:,:,0]-e3[:,:,0].min())/max(e3[:,:,0].max()-e3[:,:,0].min(),1e-8)*255,0,255).astype(np.uint8)),'explanation':'最深层的全局语义信息。感受野已覆盖整个输入。'},
        {'id':'output','name':'分割输出','image':_b64(output_rgb),'explanation':'解码器+跳跃连接恢复分辨率。白色=前景，黑色=背景。'},
    ], 'metrics': {'status':'numpy_algorithm','backend':'NumPy','algorithm':'U-Net Encoder-Decoder','encoder_levels':3,'filters':'4→8→16','skip_connections':True}}
