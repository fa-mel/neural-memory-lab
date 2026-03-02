import numpy as np


def load_mnist_patterns():
    """
    Load MNIST via TensorFlow and return binarized prototypes for digits 0-9.
    Returns a dict {digit: binary_pattern}.
    """
    import tensorflow as tf

    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train_binary = np.where(
        x_train.reshape((x_train.shape[0], -1)) > 127, 1, -1
    ).astype(np.float64)

    patterns = {}
    for digit in range(10):
        idx = np.where(y_train == digit)[0][0]
        patterns[digit] = x_train_binary[idx]

    return patterns
