import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave


def circle(size):
    r = min(size) / 4
    x, y = np.indices(size)
    x = x - r
    y = y - size[1] / 2
    return np.abs(np.hypot(r - x, y) < r).astype(int)


imsave('koleczko.png', circle((256, 512)))


def chessboard(sin1, sin2, size):
    r1, r2 = np.meshgrid(sin1, sin2)

    return np.digitize(np.cos(r1 + r2), bins=np.linspace(-1,1, num=2**size))


x = np.linspace(0, 8 * np.math.pi, num=256)
y = np.linspace(0, 8 * np.math.pi, num=256)
sin1 = np.sin(x)
sin2 = np.sin(y)
imsave('matrix.png', chessboard(sin1, sin2,2 ))
