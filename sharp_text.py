from matplotlib import pyplot as plt

import numpy as np
import pytesseract
import cv2

from nltk.corpus import words

FIND_EDGES_IMAGE_HEIGHT = 1000

def resize_image(img, new_height):
    """Resizes source image to provided max height maintaining aspect ratio"""
    (height, width) = img.shape[:2]
    ratio = new_height / float(height)
    dim = (int(width * ratio), new_height)
    return (cv2.resize(img, dim, interpolation=cv2.INTER_AREA), ratio)

image = cv2.imread("roi_0.jpg")
roi1 = image[5:80, 10:-90]
roi0 = cv2.resize(roi1, None, fx=3, fy=3)
# Define the sharpening kernel
kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])

# Apply the kernel to the image
sharpened_image = cv2.filter2D(roi0, -1, kernel)
test = cv2.fastNlMeansDenoisingColored(sharpened_image, None, 10, 10, 7, 15)

kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(test, kernel, iterations = 1)
test2=cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)
test3=cv2.threshold(test2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
test4= 255 - test3
karnel2= np.ones((3,3),np.uint8)
er = cv2.erode(test4, karnel2, iterations=1)
# Set the tessedit_char_whitelist configuration parameter
black_chars = ["(", ")", "(", ")", "="]
config = '--psm 8'
text = pytesseract.image_to_string(er, config=config)
print("Detected Number is:",text)
text_words = text.split(' ')
string_list = []
for word in text_words:
    if word.lower() in words.words() or len(word)>2:
        string_list.append(word)
final_text = ' '.join([str(elem) for elem in string_list])
print(final_text)


fig = plt.figure(figsize=(15, 10))

fig.add_subplot(2, 2, 1)
plt.imshow(roi0, cmap="gray")
plt.axis("off")
plt.title("Original image")


fig.add_subplot(2, 2, 2)
plt.imshow(test, cmap="gray")
plt.axis("off")
plt.title("Sharp")

fig.add_subplot(2, 2, 3)
plt.imshow(er, cmap="gray")
plt.axis("off")
plt.title("erosion")

fig.add_subplot(2, 2, 4)
plt.imshow(test4, cmap="gray")
plt.axis("off")
plt.title("test3")
plt.show()
