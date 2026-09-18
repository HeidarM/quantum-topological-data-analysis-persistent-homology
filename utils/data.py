# utils/data.py

# Data generation

import numpy as np


def circle(n, radius=1.0, noise=0.0, center=(0.0, 0.0)):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    data = radius * np.column_stack((np.cos(angles), np.sin(angles)))

    return np.asarray(center, dtype=float) + data + np.random.normal(0, noise, data.shape)


def translate(data, vector):
    return np.asarray(data, dtype=float) + np.asarray(vector, dtype=float)
