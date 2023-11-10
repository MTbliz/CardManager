import numpy as np
import pytesseract
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('images/img_13.png')
new_image= img.copy()
# if background is light
img = 255 - img
assert img is not None, "file could not be read, check with os.path.exists()"
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

counter = 0
# Draw the contours with hierarchy 0
new_contours = []
for i, contour in enumerate(contours):
    if hierarchy[0][i][3] == -1 :
        new_contours.append(contour)
        cv.drawContours(new_image, contours, i, (0, 255, 0), 3)
max_area = max(new_contours, key = cv.contourArea)
max_area = cv.contourArea(max_area)

for j, cnt in enumerate(new_contours):
    area = cv.contourArea(cnt)
    if max_area * 0.7 <= area <= max_area * 1.3:
        x, y, w, h = cv.boundingRect(cnt)
        cropped = new_image[y:y + h, x:x + w]
        cv.imwrite(f'cropped_{j}.jpg', cropped)

        # process text
        roi = cropped[20:80, 0::]
        ret, thresh2 = cv.threshold(roi, 127, 255, cv.THRESH_BINARY)
        #kernel = np.ones((1, 1), np.uint8)
        #erosion = cv.erode(thresh2, kernel, iterations=1)
        cv.imwrite(f'roi_{j}.jpg', thresh2)
        data = pytesseract.image_to_data(thresh2, output_type=pytesseract.Output.DICT)
        words_vote = list(zip(data['text'], data['conf']))
        # Filter the list of recognized words
        words = [word[0] for word in words_vote if int(word[1]) > 50]
        final_string = ' '.join(words).strip()
        print(final_string)


#cv.drawContours(img, contours, -1, (0,255,0), 3)
plt.subplot(122),plt.imshow(new_image)
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()