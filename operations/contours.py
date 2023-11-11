import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('img.png')
new_image= img.copy()
img = 255 - img;
assert img is not None, "file could not be read, check with os.path.exists()"
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

counter = 0
# Draw the contours with hierarchy 0
for i, contour in enumerate(contours):
  if hierarchy[0][i][3] == -1 :
     cv.drawContours(new_image, contours, i, (0, 255, 0), 3)

#cv.drawContours(img, contours, -1, (0,255,0), 3)
plt.subplot(122),plt.imshow(new_image)
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()