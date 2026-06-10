"""Simplified GAN demo. Pure NumPy."""
import numpy as np


class SimpleGAN:
    """
    Minimal GAN for educational visualization.
    Generator: small MLP (noise -> fake image)
    Discriminator: small MLP (image -> real/fake score)
    """

    def __init__(self, noise_dim=10, image_dim=784):
        self.noise_dim = noise_dim
        self.image_dim = image_dim
        rng = np.random.default_rng(42)

        # Generator: noise -> hidden -> image
        self.G_W1 = rng.normal(0, 0.1, (64, noise_dim)).astype(np.float64)
        self.G_b1 = np.zeros(64, dtype=np.float64)
        self.G_W2 = rng.normal(0, 0.1, (image_dim, 64)).astype(np.float64)
        self.G_b2 = np.zeros(image_dim, dtype=np.float64)

        # Discriminator: image -> hidden -> score
        self.D_W1 = rng.normal(0, 0.1, (64, image_dim)).astype(np.float64)
        self.D_b1 = np.zeros(64, dtype=np.float64)
        self.D_W2 = rng.normal(0, 0.1, (1, 64)).astype(np.float64)
        self.D_b2 = np.zeros(1, dtype=np.float64)

    def generate(self, noise=None):
        """Generate fake images from noise."""
        if noise is None:
            rng = np.random.default_rng()
            noise = rng.normal(0, 1, (1, self.noise_dim)).astype(np.float64)
        z = np.asarray(noise, dtype=np.float64).reshape(-1, self.noise_dim)
        h = np.tanh(z @ self.G_W1.T + self.G_b1)
        fake = np.tanh(h @ self.G_W2.T + self.G_b2)
        return fake

    def discriminate(self, images):
        """Score images (real=1, fake=0)."""
        x = np.asarray(images, dtype=np.float64).reshape(-1, self.image_dim)
        h = np.tanh(x @ self.D_W1.T + self.D_b1)
        score = 1.0 / (1.0 + np.exp(-(h @ self.D_W2.T + self.D_b2)))  # sigmoid
        return score

    def train_step(self, real_images, lr=0.01):
        """One step of GAN training."""
        batch_size = real_images.shape[0] if real_images.ndim > 1 else 1
        x_real = np.asarray(real_images, dtype=np.float64).reshape(batch_size, -1)

        rng = np.random.default_rng()
        noise = rng.normal(0, 1, (batch_size, self.noise_dim)).astype(np.float64)
        x_fake = self.generate(noise)

        # Train discriminator
        real_score = self.discriminate(x_real)
        fake_score = self.discriminate(x_fake)
        d_loss = -np.mean(np.log(real_score + 1e-8) + np.log(1.0 - fake_score + 1e-8))

        # Simplified gradient update for D (conceptual)
        # Real gradients push score towards 1, fake towards 0
        d_grad_real = (real_score - 1.0) / batch_size
        d_grad_fake = (1.0 - fake_score) / batch_size

        self.D_W1 -= lr * np.outer(d_grad_real, x_real.mean(axis=0)) @ self.D_W1 * 0.1
        self.D_W1 -= lr * np.outer(d_grad_fake, x_fake.mean(axis=0)) @ self.D_W1 * 0.1

        # Train generator (make discriminator believe fakes are real)
        g_loss = -np.mean(np.log(fake_score + 1e-8))
        g_grad = (1.0 - fake_score) / batch_size
        self.G_W1 += lr * np.outer(g_grad, noise.mean(axis=0)) @ self.G_W1 * 0.1

        return float(d_loss), float(g_loss)
