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
s = 1
T_shear = np.array([
    [1, s, 0],
    [0, 1, 0],
    [0, 0, 1],
])

T_rotate = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]])

scale = 4
T_scale = np.array([
    [scale, 0, 0],
    [0, scale, 0],
    [0, 0, 1]])

img = imread('./lenna.png', as_gray=True)
imgint = img_as_uint(img)
height = len(imgint)
width = len(imgint[0])

x = np.linspace(0, width-1, num=width).astype(int)
y = np.linspace(0, height-1, num=height).astype(int)

mesh = np.array(np.meshgrid(x, y, 1)).T
# print(mesh)


res1 = mesh @ T_scale
res2 = mesh @ T_shear
res3 = mesh @ T_rotate

print(res1[0], len(res1[0][1][0]))
# print(res2)
# print(res3)

resx = res1[0][0]
resy = res1[0][1]
resx2 = res2[0][0]
resy2 = res2[0][1]
resx3 = res3[0][0]
resy3 = res3[0][1]

imgres1 = np.zeros((width, height))
imgres2 = np.zeros((width, height))
imgres3 = np.zeros((width, height))

inx = res1[0][0 : width][0 : height] % width
ix = inx[:, :, 0]
iy = inx[:,:, 1]

inx2 = res2[0][0 : width][0 : height] % width
ix2 = inx2[:, :, 0]
iy2 = inx2[:,:, 1]

inx3 = res3[0][0 : width][0 : height] % width
ix3 = inx3[:, :, 0]
iy3 = inx3[:,:, 1]

imgres1 = imgint[ix,iy]
# print(imgres1)
imgres2 = imgint[ix2,iy2]
imgres3 = imgint[ix3,iy3]


imsave('test.png', imgint)
imsave('test1.png', imgres1.astype(int))
imsave('test2.png', imgres2.astype(int))
imsave('test3.png', imgres3.astype(int))

# zaszumianie obrazu, wygenerowac z szumem i glebia kilka obrazów, potem odszumic, jedna operacja matematyczna
# przeksztalcenia liniowe, negatyw, losowe (zamiana pikseli)
# wyrysowac wektor do przekstalcenia i sam obraz po przeksztalceniu