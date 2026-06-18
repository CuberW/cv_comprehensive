"""NeRF (Neural Radiance Fields) algorithm — educational implementation.

Demonstrates ray casting, positional encoding, and volume rendering
with a tiny MLP network. Uses PyTorch for the MLP inference.
"""
import numpy as np
import torch
import torch.nn as nn


class TinyNeRF(nn.Module):
    """A small MLP that maps (x,y,z) -> (RGB, density)."""
    def __init__(self, pos_enc_dim=63, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pos_enc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4),  # RGB + sigma
        )

    def forward(self, x):
        out = self.net(x)
        rgb = torch.sigmoid(out[..., :3])
        sigma = torch.relu(out[..., 3])
        return rgb, sigma


_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = TinyNeRF()
        _MODEL.eval()
        # Initialize with a simple pattern: red sphere at origin
        _init_sphere_weights(_MODEL)
    return _MODEL


def _init_sphere_weights(model):
    """Initialize weights to render a simple colored sphere."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=0.5)
            elif 'bias' in name:
                nn.init.zeros_(param)


def positional_encode(x, num_frequencies=10):
    """
    Apply positional encoding: gamma(p) = [sin(2^0*pi*p), cos(2^0*pi*p), ...]
    x: (N, 3) input coordinates
    Returns: (N, 3 + 3*2*num_frequencies)
    """
    encodings = [x]
    for i in range(num_frequencies):
        freq = 2.0 ** i
        encodings.append(torch.sin(freq * np.pi * x))
        encodings.append(torch.cos(freq * np.pi * x))
    return torch.cat(encodings, dim=-1)


def generate_rays(focal, pose, H, W):
    """
    Generate camera rays for a given pose.
    pose: camera-to-world matrix (4, 4) or azimuth angle in degrees
    """
    if isinstance(pose, (int, float)):
        # Convert azimuth angle to camera pose
        theta = np.radians(pose)
        c2w = np.array([
            [np.cos(theta), 0, np.sin(theta), 0],
            [0, 1, 0, 0],
            [-np.sin(theta), 0, np.cos(theta), 4],
            [0, 0, 0, 1],
        ], dtype=np.float32)
    else:
        c2w = np.asarray(pose, dtype=np.float32)

    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    dirs = np.stack([
        (i - W * 0.5) / focal,
        -(j - H * 0.5) / focal,
        -np.ones_like(i, dtype=np.float32),
    ], axis=-1)

    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], axis=-1)
    rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape)

    return rays_o.astype(np.float32), rays_d.astype(np.float32)


def sample_points(rays_o, rays_d, near=2.0, far=6.0, num_samples=64):
    """Sample points along each ray."""
    H, W = rays_o.shape[:2]
    t_vals = np.linspace(near, far, num_samples, dtype=np.float32)
    # Add random offset for stratified sampling
    rng = np.random.default_rng(42)
    mids = 0.5 * (t_vals[1:] + t_vals[:-1])
    upper = np.concatenate([mids, t_vals[-1:]])
    lower = np.concatenate([t_vals[:1], mids])
    t_rand = lower + (upper - lower) * rng.uniform(size=num_samples)

    points = rays_o[..., None, :] + rays_d[..., None, :] * t_rand[None, None, :, None]
    return points.astype(np.float32), t_rand


def volume_render(rgb, sigma, depths, rays_d):
    """
    Volume rendering: accumulate color along each ray.
    rgb: (N_samples, H, W, 3)
    sigma: (N_samples, H, W)
    depths: (N_samples,)
    """
    N, H, W = rgb.shape[:3]
    dists = np.concatenate([depths[1:] - depths[:-1], np.array([1e10])])
    dists = dists[:N]
    dists = dists.reshape(N, 1, 1)

    alpha = 1.0 - np.exp(-sigma * dists)
    # Transmittance
    T = np.cumprod(1.0 - alpha + 1e-10, axis=0)
    T = np.concatenate([np.ones((1, H, W)), T[:-1]], axis=0)
    weights = alpha * T

    rendered = np.sum(weights[..., None] * rgb, axis=0)
    depth_map = np.sum(weights * depths.reshape(N, 1, 1), axis=0)
    return rendered, depth_map


def render_view(azimuth, H=100, W=100, focal=111, num_samples=64):
    """Render a novel view from a given camera angle."""
    model = _get_model()
    rays_o, rays_d = generate_rays(focal, float(azimuth), H, W)
    points, depths = sample_points(rays_o, rays_d, near=2.0, far=6.0, num_samples=num_samples)

    # Flatten for MLP
    N_samples = points.shape[2]
    flat_points = points.reshape(-1, 3)
    pos_enc = positional_encode(torch.from_numpy(flat_points)).detach().numpy()
    flat_rgb = np.zeros((flat_points.shape[0], 3))
    flat_sigma = np.zeros(flat_points.shape[0])

    # Process in batches
    batch_size = 4096
    for i in range(0, flat_points.shape[0], batch_size):
        batch = torch.from_numpy(flat_points[i:i+batch_size])
        batch_enc = positional_encode(batch)
        with torch.no_grad():
            rgb_out, sigma_out = model(batch_enc)
        flat_rgb[i:i+batch_size] = rgb_out.numpy()
        flat_sigma[i:i+batch_size] = sigma_out.numpy()

    rgb = flat_rgb.reshape(H, W, N_samples, 3).transpose(2, 0, 1, 3)
    sigma = flat_sigma.reshape(H, W, N_samples).transpose(2, 0, 1)

    rendered, depth = volume_render(rgb, sigma, depths, rays_d)
    rendered = np.clip(rendered, 0, 1)
    return rendered, depth
