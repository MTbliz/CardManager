import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('fab_cards.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
#ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
edges = cv.Canny(img,100,250)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()