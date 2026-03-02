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
    flip_indices = np.random.choice(len(noisy), size=num_to_flip, replace=False)
    noisy[flip_indices] *= -1
    return noisy


def binarize(image_array):
    flat = np.asarray(image_array).flatten()
    return np.where(flat > 127, 1.0, -1.0).astype(np.float64)


def pattern_to_image(pattern):
    arr = np.asarray(pattern, dtype=np.float64).flatten()
    return ((arr.reshape(28, 28) + 1) / 2 * 255).astype(np.uint8)
