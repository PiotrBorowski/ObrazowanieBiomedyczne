import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_uint
from skimage.io import imsave, imread


# ZAD 1

def square(size, side, start):
    image = np.zeros((size, size)).astype(np.uint8)
    image[start[0]:start[0] + side, start[1]:start[1] + side] = 255
    return image


imsave('square.png', square(512, 123, (32,64)))


def midcircle(size):
    r = min(size) / 4
    x, y = np.indices(size)
    x = x - size[0] / 2 + r
    y = y - size[1] / 2
    return np.abs(np.hypot(r - x, y) < r).astype(int)


imsave('koleczko1.png', midcircle((256, 512)))
imsave('koleczko2.png', midcircle((512, 256)))


# ZAD 2

def checkerboard(sin1, sin2, size):
    r1, r2 = np.meshgrid(sin1, sin2)
    return np.digitize(r1*r2, bins=np.linspace(-1,1, num=2**size+1))


x = np.linspace(0, 8 * np.math.pi, num=256)
y = np.linspace(0, 8 * np.math.pi, num=256)
sin1 = np.sin(x)
sin2 = np.sin(y)
imsave('matrix.png', checkerboard(sin1, sin2, 2))

plt.subplot(221)
plt.plot(x, sin1)

plt.subplot(222)
plt.imshow(checkerboard(sin1, sin2, 8), cmap='gray')

plt.subplot(223)
plt.imshow(checkerboard(sin1, sin2, 3), cmap='gray')

plt.show()


# ZAD 3
T_reflect = np.array([
    [1,0,0],
    [0,-1,0],
    [0,0,0]
])

s = np.sqrt(3)
T_shear = np.array([
    [1, s, 0],
    [0, 1, 0],
    [0, 0, 1],
])

T_rotate = np.array([
    [np.sqrt(2)/2, np.sqrt(2)/2, 0],
    [np.sqrt(2)/-2, np.sqrt(2)/2, 0],
    [0, 0, 1]])

scale = 1.7
T_scale = np.array([
    [scale, 0, 0],
    [0, scale, 0],
    [0, 0, 1]])

T_pos = np.array([
    [1, 0, 512],
    [0, 1, 512],
    [0, 0, 1]])

T_neg = np.array([
    [1, 0, -250],
    [0, 1, -250],
    [0, 0, 1]])

img = imread('./lenna.png', as_gray=True)

def affineTransform(img, transform):
    imgint = img_as_uint(img)
    height = len(imgint)
    width = len(imgint[0])
    x = np.linspace(0, width - 1, num=width).astype(int)
    y = np.linspace(0, height - 1, num=height).astype(int)
    mesh = np.array(np.meshgrid(x, y, 1)).T
    res = mesh @ transform
    res = res.astype(int)
    inx = res[0][:][:] % width
    ix = inx[:, :, 0]
    iy = inx[:,:, 1]
    imgres = np.zeros((max(inx.flatten()), max(inx.flatten())))
    imgres = imgint[ix,iy]
    return imgres;



imsave('shear.png', affineTransform(img, T_shear))
imsave('rotate.png', affineTransform(img, T_rotate))
imsave('scale.png', affineTransform(img, T_scale))
imsave('reflect.png', affineTransform(img, T_reflect))

# zaszumianie obrazu, wygenerowac z szumem i glebia kilka obrazów, potem odszumic, jedna operacja matematyczna
# przeksztalcenia liniowe, negatyw, losowe (zamiana pikseli)
# wyrysowac wektor do przekstalcenia i sam obraz po przeksztalceniu