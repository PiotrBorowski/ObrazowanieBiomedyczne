import numpy as np
import matplotlib.pyplot as plt
from numpy import uint8, array
from sklearn.neighbors._dist_metrics import DistanceMetric

size = (4, 4)

image = np.zeros(size).astype(np.uint8)
image
# np.array([[0, 0, 0, 0],
#           [0, 0, 0, 0],
#           [0, 0, 0, 0],
#           [0, 0, 0, 0]], dtype=np.uint8)

image[2, 3] = 128
image
# np.array([[0, 0, 0, 0],
#           [0, 0, 0, 0],
#           [0, 0, 0, 128],
#           [0, 0, 0, 0]], dtype=np.uint8)

image[1: 3, 0: 2] = 64
image
# np.array([[0, 0, 0, 0],
#           [64, 64, 0, 0],
#           [64, 64, 0, 128],
#           [0, 0, 0, 0]], dtype=np.uint8)

image > 32
# np.array([[False, False, False, False],
#           [True, True, False, False],
#           [True, True, False, True],
#           [False, False, False, False]])

image[image > 32] = 32
image
# np.array([[0, 0, 0, 0],
#           [32, 32, 0, 0],
#           [32, 32, 0, 32],
#           [0, 0, 0, 0]], dtype=np.uint8)

image[0, 0] = 300
image
# np.array([[44, 0, 0, 0],
#           [32, 32, 0, 0],
#           [32, 32, 0, 32],
#           [0, 0, 0, 0]], dtype=np.uint8)

image = np.zeros((128, 128)).astype(np.uint8)
image[1:64, 1:64] = 255
image
# array([[0, 0, 0, ..., 0, 0, 0],
#        [0, 255, 255, ..., 0, 0, 0],
#        [0, 255, 255, ..., 0, 0, 0],
#        ...,
#        [0, 0, 0, ..., 0, 0, 0],
#        [0, 0, 0, ..., 0, 0, 0],
#        [0, 0, 0, ..., 0, 0, 0]], dtype=object)

plt.imsave('foo.png', image, cmap="gray")

image = np.zeros((256, 256)).astype(np.uint8)
for i in range(256):
    image[:, i] = i * 2
plt.imsave('bar.png', image, cmap="gray")

# ------------------------------------------------------------
def circle(r):
    d = 2 * r
    rx, ry = d / 2, d / 2
    x, y = np.indices((2*d, 2*d))
    x = x - r
    y = y - r
    print(x, y)
    return (np.abs(np.hypot(rx - x, ry - y) - r) > 1).astype(int)


plt.imsave('kolko.png', circle(256) , cmap="gray")

# rysowanie bez petli kolka numpy, korzystanie z distance metric
# numpy złozyc z dwoch sinsow jeden, obraz z glebia linespace, min, max, potegowanie
# przeksztalcenie afiniczne, wczytac i zrobic przeksztalcenia
