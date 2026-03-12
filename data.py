"""Load and binarize MNIST prototypes (one per digit class)."""
import numpy as np


def load_mnist_patterns():
    """
    Returns dict  {digit: binary_pattern}  for digits 0-9.
    Each pattern is a 784-element float64 array with values ±1.
    """
    try:
        import tensorflow as tf
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    except ImportError:
        raise ImportError(
            "TensorFlow is required to load MNIST.  "
            "Install it with:  pip install tensorflow"
        )

    x_bin = np.where(
        x_train.reshape(x_train.shape[0], -1) > 127, 1, -1
    ).astype(np.float64)

    patterns = {}
    for d in range(10):
        idx = np.where(y_train == d)[0][0]
        patterns[d] = x_bin[idx]
    return patterns

