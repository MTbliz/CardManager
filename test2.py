from matplotlib import pyplot as plt
import numpy as np
import pytesseract
import cv2

class ReaderError(Exception):
    """Meter reader generic exception"""

img_path = "images/img_4.png"

# Load the input image and convert to RGB for correct display
image = cv2.imread(img_path)
image_ini = image.copy()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
orig = image.copy()

FIND_EDGES_IMAGE_HEIGHT = 1000

def resize_image(img, new_height):
    """Resizes source image to provided max height maintaining aspect ratio"""
    (height, width) = img.shape[:2]
    ratio = new_height / float(height)
    dim = (int(width * ratio), new_height)
    return (cv2.resize(img, dim, interpolation=cv2.INTER_AREA), ratio)

def get_edge_detection_thresholds(img):
    """Calculates the lower and upper thresholds for Canny edge detection"""
    sigma = 0.3
    median = np.median(img)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return (lower, upper)

find_display_image = orig.copy()
find_display_image = cv2.cvtColor(find_display_image, cv2.COLOR_RGB2GRAY)

# Resize the image for more accurate contour detection
(find_display_image, ratio) = resize_image(find_display_image, FIND_EDGES_IMAGE_HEIGHT)
(new_image, ratio) = resize_image(image_ini, FIND_EDGES_IMAGE_HEIGHT)

# Apply blur and detect edges
find_display_image = cv2.GaussianBlur(find_display_image, (3, 3), 0)
(lower, upper) = get_edge_detection_thresholds(find_display_image)
print(lower, upper)
edged = cv2.Canny(find_display_image, lower, upper)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dilated = cv2.dilate(edged, kernel, iterations=2)
contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours with hierarchy 0
new_contours = []
for i, contour in enumerate(contours):
    if hierarchy[0][i][3] == -1 :
        new_contours.append(contour)
        cv2.drawContours(new_image, contours, i, (0, 255, 0), 3)
max_area = max(new_contours, key = cv2.contourArea)
max_area = cv2.contourArea(max_area)
for j, cnt in enumerate(new_contours):
    area = cv2.contourArea(cnt)
    if max_area * 0.5 <= area <= max_area * 1.5:
        x, y, w, h = cv2.boundingRect(cnt)
        cropped = new_image[y:y + h, x:x + w]
        cv2.imwrite(f'cropped_{j}.jpg', cropped)




       # process text
        roi0 = cropped[20:100, 0::]
        (roi, ratio) = resize_image(roi0, FIND_EDGES_IMAGE_HEIGHT)
        ret, thresh2 = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)
        #kernel = np.ones((1, 1), np.uint8)
        #erosion = cv.erode(thresh2, kernel, iterations=1)
        cv2.imwrite(f'roi_{j}.jpg', roi0)
        data = pytesseract.image_to_data(roi, output_type=pytesseract.Output.DICT)
        words_vote = list(zip(data['text'], data['conf']))
        # Filter the list of recognized words
        words = [word[0] for word in words_vote if int(word[1]) > 50]
        final_string = ' '.join(words).strip()
        print(final_string)






fig = plt.figure(figsize=(15, 10))

fig.add_subplot(2, 2, 1)
plt.imshow(orig, cmap="gray")
plt.axis("off")
plt.title("Original image")

fig.add_subplot(2, 2, 2)
plt.imshow(find_display_image, cmap="gray")
plt.axis("off")
plt.title("Unedged image")

fig.add_subplot(2, 2, 3)
plt.imshow(edged, cmap="gray")
plt.axis("off")
plt.title("Edged image")

fig.add_subplot(2, 2, 4)
plt.imshow(dilated, cmap="gray")
plt.axis("off")
plt.title("Dilated image")
plt.show()