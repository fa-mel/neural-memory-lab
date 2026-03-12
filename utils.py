"""Utility functions for pattern manipulation and complexity metrics."""
import numpy as np
from PIL import Image


def overlap(s1, s2):
    """Macroscopic overlap O = (1/N) Σ s1_i · s2_i ."""
    a = np.asarray(s1, dtype=np.float64).flatten()
    b = np.asarray(s2, dtype=np.float64).flatten()
    return float(np.dot(a, b) / len(a))


def add_noise(pattern, fraction, seed=None):
    """Flip a fraction of spins uniformly at random."""
    if seed is not None:
        np.random.seed(seed)
    noisy = np.asarray(pattern, dtype=np.float64).flatten().copy()
    n_flip = int(fraction * len(noisy))
    if n_flip > 0:
        idx = np.random.choice(len(noisy), size=n_flip, replace=False)
        noisy[idx] *= -1
    return noisy


def binarize(image_array):
    """Convert grayscale 0-255 to bipolar ±1."""
    flat = np.asarray(image_array).flatten()
    return np.where(flat > 127, 1.0, -1.0).astype(np.float64)


def pattern_to_image(pattern, size=28):
    """Convert ±1 vector → uint8 array suitable for PIL."""
    arr = np.asarray(pattern, dtype=np.float64).flatten()
    img = ((arr.reshape(size, size) + 1) / 2 * 255).astype(np.uint8)
    return img


def to_pil(pattern, display_size=168):
    """Pattern → PIL Image, upscaled with nearest-neighbor (crisp pixels)."""
    arr = pattern_to_image(pattern)
    return Image.fromarray(arr, mode="L").resize(
        (display_size, display_size), Image.NEAREST
    )


def sample_entropy(time_series, m=2, r_fraction=0.2):
    """
    Sample Entropy (SampEn) via Chebyshev distance.
    Quantifies regularity of the energy trajectory.
    """
    ts = np.asarray(time_series, dtype=np.float64)
    nonzero = np.where(ts != 0)[0]
    if len(nonzero) < 10:
        return 0.0
    ts = ts[: nonzero[-1] + 1]
    N = len(ts)
    if N < m + 2:
        return 0.0
    std = np.std(ts)
    if std == 0:
        return 0.0
    r = r_fraction * std

    tpl_m = np.array([ts[i : i + m] for i in range(N - m)])
    tpl_m1 = np.array([ts[i : i + m + 1] for i in range(N - m)])

    B = np.sum(np.all(np.abs(tpl_m[:, None] - tpl_m[None, :]) <= r, axis=-1)) - len(tpl_m)
    A = np.sum(np.all(np.abs(tpl_m1[:, None] - tpl_m1[None, :]) <= r, axis=-1)) - len(tpl_m1)

    if A > 0 and B > 0:
        return float(-np.log(A / B))
    return 0.0


def frames_to_gif(frames, duration_ms=60):
    """Convert list of PIL images to in-memory GIF bytes."""
    import io
    buf = io.BytesIO()
    frames[0].save(
        buf, format="GIF", save_all=True,
        append_images=frames[1:], loop=0, duration=duration_ms,
    )
    buf.seek(0)
    return buf.read()
