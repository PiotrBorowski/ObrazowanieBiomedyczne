from skimage import img_as_uint
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt

img = imread('./lenna.png', as_gray=True)
imgint = img_as_uint(img).astype(np.uint8)

np.random.seed(0);


noised = imgint + np.random.randint(-2**11, 2**11, (1000, imgint.shape[0], imgint.shape[1]));

denoised = np.average(noised, axis=0)


f, axarr = plt.subplots(3,3)
axarr[0,0].imshow(noised[0])
axarr[0,1].imshow(noised[1])
axarr[0,2].imshow(noised[2])

axarr[1,0].imshow(noised[3])
axarr[1,1].imshow(noised[4])
axarr[1,2].imshow(noised[5])

axarr[2,0].imshow(noised[6])
axarr[2,1].imshow(noised[7])
axarr[2,2].imshow(noised[8])

plt.show()


f, axarr = plt.subplots(1,2)

axarr[0].imshow(imgint)
axarr[1].imshow(denoised)


# 2, 3

identity = np.arange(256)
negative = np.linspace(255, 0, 256).astype(np.uint8)
rand = np.random.randint(256, size=256)

edge = np.ones(256);
dev = np.std(identity).astype(np.uint8)
avg = np.average(identity).astype(np.uint8)
edge[avg - dev : avg + dev] = 0


imgid = identity[imgint]
imgneg = negative[imgint]
imgrand = rand[imgint]
imgedge = edge[imgint]

f, axarr = plt.subplots(4,2)

axarr[0,0].plot(identity, identity)
axarr[0,1].imshow(imgid)

axarr[1,0].plot(identity, negative)
axarr[1,1].imshow(imgneg)

axarr[3,0].plot(identity, edge)
axarr[3,1].imshow(imgedge)

axarr[2,0].plot(identity, rand)
axarr[2,1].imshow(imgrand)


plt.show()

# 3
log = np.log(identity)
imglog = log[imgint]


gamma1 = np.array(255 * (identity / 255) ** 0.1, dtype='uint8')
gamma5 = np.array(255 * (identity / 255) ** 0.5, dtype='uint8')
gamma15 = np.array(255 * (identity / 255) ** 1.5, dtype='uint8')
gamma25 = np.array(255 * (identity / 255) ** 2.5, dtype='uint8')

imggamma1 = gamma1[imgint]
imggamma5 = gamma5[imgint]
imggamma15 = gamma15[imgint]
imggamma25 = gamma25[imgint]

f, axarr = plt.subplots(6,2)

axarr[0,0].plot(identity, identity)
axarr[0,1].imshow(imgid)

axarr[1,0].plot(identity, log)
axarr[1,1].imshow(imglog)

axarr[2,0].plot(identity, gamma1)
axarr[2,1].imshow(imggamma1)

axarr[3,0].plot(identity, gamma5)
axarr[3,1].imshow(imggamma5)

axarr[4,0].plot(identity, gamma15)
axarr[4,1].imshow(imggamma15)

axarr[5,0].plot(identity, gamma25)
axarr[5,1].imshow(imggamma25)

plt.show()










