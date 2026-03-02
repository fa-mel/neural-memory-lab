import numpy as np


def calculate_overlap(state1, state2):
    return np.dot(state1, state2) / len(state1)


def add_noise(pattern, noise_level, seed=None):
    if seed is not None:
        np.random.seed(seed)
    noisy = pattern.copy()
    num_to_flip = int(noise_level * len(pattern))
    flip_indices = np.random.choice(len(pattern), size=num_to_flip, replace=False)
    noisy[flip_indices] *= -1
    return noisy


def binarize(image_array):
    flat = image_array.flatten()
    return np.where(flat > 127, 1, -1).astype(np.float64)


def pattern_to_image(pattern):
    return ((pattern.reshape(28, 28) + 1) / 2 * 255).astype(np.uint8)
