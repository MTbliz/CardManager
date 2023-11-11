import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('fab_cards.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
kernel = np.ones((5,5),np.uint8)
ret,thresh1 = cv.threshold(img,225,255,cv.THRESH_BINARY)
closing = cv.morphologyEx(thresh1, cv.MORPH_CLOSE, kernel)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(thresh1),plt.title('Erosion')
plt.xticks([]), plt.yticks([])
plt.show()