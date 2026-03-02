import numpy as np


def calculate_overlap(state1, state2):
    s1 = np.asarray(state1, dtype=np.float64).flatten()
    s2 = np.asarray(state2, dtype=np.float64).flatten()
    return float(np.dot(s1, s2) / len(s1))


def add_noise(pattern, noise_level, seed=None):
    if seed is not None:
        np.random.seed(seed)
    noisy = np.asarray(pattern, dtype=np.float64).flatten().copy()
    num_to_flip = int(noise_level * len(noisy))
    if num_to_flip > 0:
        flip_indices = np.random.choice(len(noisy), size=num_to_flip, replace=False)
        noisy[flip_indices] *= -1
    return noisy


def binarize(image_array):
    flat = np.asarray(image_array).flatten()
    return np.where(flat > 127, 1.0, -1.0).astype(np.float64)


def pattern_to_image(pattern):
    arr = np.asarray(pattern, dtype=np.float64).flatten()
    return ((arr.reshape(28, 28) + 1) / 2 * 255).astype(np.uint8)


def calculate_sampen(time_series, m=2, r_fraction=0.2):
    """Sample Entropy via Chebyshev distance — quantifies energy trajectory complexity."""
    ts = np.asarray(time_series, dtype=np.float64)
    nonzero = np.where(ts != 0)[0]
    if len(nonzero) < 10:
        return 0.0
    ts = ts[:nonzero[-1] + 1]
    N = len(ts)
    if N < m + 2:
        return 0.0
    std_dev = np.std(ts)
    if std_dev == 0:
        return 0.0
    r = r_fraction * std_dev

    templates_m  = np.array([ts[i:i + m]     for i in range(N - m)])
    templates_m1 = np.array([ts[i:i + m + 1] for i in range(N - m)])

    diff_m  = np.abs(templates_m[:, None, :]  - templates_m[None, :, :])
    B = np.sum(np.all(diff_m  <= r, axis=-1)) - len(templates_m)

    diff_m1 = np.abs(templates_m1[:, None, :] - templates_m1[None, :, :])
    A = np.sum(np.all(diff_m1 <= r, axis=-1)) - len(templates_m1)

    if A > 0 and B > 0:
        return float(-np.log(A / B))
    return 0.0
