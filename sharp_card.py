from matplotlib import pyplot as plt
import numpy as np
import pytesseract
import cv2

def resize_image(img, new_height):
    """Resizes source image to provided max height maintaining aspect ratio"""
    (height, width) = img.shape[:2]
    ratio = new_height / float(height)
    dim = (int(width * ratio), new_height)
    return (cv2.resize(img, dim, interpolation=cv2.INTER_AREA), ratio)

def get_edge_detection_thresholds(img):
    """Calculates the lower and upper thresholds for Canny edge detection"""
    sigma = 0.9
    median = np.median(img)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return (lower, upper)

image = cv2.imread('cropped_5.jpg')
(resized_image, ratio) = resize_image(image, 2000)
top_image = resized_image[:resized_image.shape[0]//5]
new_image = top_image.copy()


# Apply blur and detect edges
find_display_image = cv2.GaussianBlur(top_image, (3, 3), 0)
(lower, upper) = get_edge_detection_thresholds(find_display_image)
print(lower, upper)
edged = cv2.Canny(find_display_image, lower, upper)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dilated = cv2.dilate(edged, kernel, iterations=1)
contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

height, width, channels = top_image.shape
top_image_are = height * width
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if top_image_are * 0.06 < area < top_image_are * 0.3:
        #cv2.drawContours(new_image, contours, i , (0, 255, 0), 3)
        x, y, w, h = cv2.boundingRect(contour)
        cropped = new_image[y:y + h, x:x + w]
        cropped_final = cropped[:,:-cropped.shape[1]//5]
        cv2.imwrite(f'sharp_{i}.jpg', cropped_final)


fig = plt.figure(figsize=(15, 10))

fig.add_subplot(2, 2, 1)
plt.imshow(top_image, cmap="gray")
plt.axis("off")
plt.title("Original image")

fig.add_subplot(2, 2, 2)
plt.imshow(new_image, cmap="gray")
plt.axis("off")
plt.title(" image to test")
plt.show()