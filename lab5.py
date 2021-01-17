from skimage import img_as_uint
from skimage.data import coins
from skimage.feature import canny
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation, binary_closing, binary_opening, selem
import numpy as np
from scipy.ndimage import binary_fill_holes
from sklearn.metrics import jaccard_score

# obraz referencyjny maski
ref = imread('./ground_truth.png', as_gray=True).astype(np.bool)

# zaczytanie obrazu z monetami
coins = coins()

# detekcja krawędzi za pomocą det. Canny
cannyed = canny(coins, high_threshold=255)

# zamknięcie z ele. strukturalnym w kształcie dysku o r = 2
mask = binary_closing(cannyed, selem=selem.disk(2))
# wypełnianie dziur
mask = binary_fill_holes(mask)

fig, ax = plt.subplots(2,2)
ax[0,0].imshow(coins, cmap='gray')
ax[0,0].set_title('coins')

ax[0,1].imshow(coins * mask, cmap='gray')
ax[0,1].set_title('coins * mask')

ax[1,0].imshow(mask, cmap='gray')
ax[1,0].set_title('mask')

ax[1,1].imshow(ref, cmap='gray')
ax[1,1].set_title('ground_truth')

# indeks Jaccarda
score = jaccard_score(mask.ravel(), np.array(ref).ravel())
print(score)

fig.show()

