import cv2
import re
import numpy as np
import pytesseract
from common import resize_image, get_edge_detection_thresholds


def detect_cards(ini_image):
    # Load the input image and convert to RGB for correct display
    image = ini_image
    image_ini = image.copy()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig = image.copy()

    FIND_EDGES_IMAGE_HEIGHT = 1000

    find_display_image = orig.copy()
    find_display_image = cv2.cvtColor(find_display_image, cv2.COLOR_RGB2GRAY)

    # Resize the image for more accurate contour detection
    (find_display_image, ratio) = resize_image(find_display_image, FIND_EDGES_IMAGE_HEIGHT)
    (new_image, ratio) = resize_image(image_ini, FIND_EDGES_IMAGE_HEIGHT)

    # Apply blur and detect edges
    find_display_image = cv2.GaussianBlur(find_display_image, (3, 3), 0)
    (lower, upper) = get_edge_detection_thresholds(find_display_image, 0.3)

    edged = cv2.Canny(find_display_image, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(edged, kernel, iterations=2)
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours with hierarchy 0
    new_contours = []
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] == -1:
            new_contours.append(contour)
            cv2.drawContours(new_image, contours, i, (0, 255, 0), 3)
    max_area = max(new_contours, key=cv2.contourArea)
    max_area = cv2.contourArea(max_area)
    detected_cards = []
    for j, cnt in enumerate(new_contours):
        area = cv2.contourArea(cnt)
        if max_area * 0.5 <= area <= max_area * 1.5:
            x, y, w, h = cv2.boundingRect(cnt)
            cropped = new_image[y:y + h, x:x + w]
            detected_cards.append(cropped)
            cv2.imwrite(f'cropped_{j}.jpg', cropped)
    return detected_cards


def get_card_title_area(image):
    (resized_image, ratio) = resize_image(image, 2000)
    top_image = resized_image[:resized_image.shape[0] // 5]
    new_image = top_image.copy()

    # Apply blur and detect edges
    find_display_image = cv2.GaussianBlur(top_image, (3, 3), 0)
    (lower, upper) = get_edge_detection_thresholds(find_display_image, 0.9)
    print(lower, upper)
    edged = cv2.Canny(find_display_image, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(edged, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    height, width, channels = top_image.shape
    top_image_are = height * width

    cards_title_areas=[]
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if top_image_are * 0.06 < area < top_image_are * 0.3:
            # cv2.drawContours(new_image, contours, i , (0, 255, 0), 3)
            x, y, w, h = cv2.boundingRect(contour)
            cropped = new_image[y:y + h, x:x + w]
            cropped_final = cropped[:, :-cropped.shape[1] // 5]
            height, width, channels = cropped_final.shape
            cropped_area = height * width
            title_with_area = (cropped_final, cropped_area)
            cards_title_areas.append(title_with_area)
            cv2.imwrite(f'sharp_{i}.jpg', cropped_final)
    if len(cards_title_areas) > 0:
        title_with_min_area = min(cards_title_areas, key=lambda tup: tup[1])
        return title_with_min_area[0]
    else:
        return None


def get_card_title(image):
    roi1 = image
    roi0 = cv2.resize(roi1, None, fx=3, fy=3)
    # Define the sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    # Apply the kernel to the image
    sharpened_image = cv2.filter2D(roi0, -1, kernel)
    test = cv2.fastNlMeansDenoisingColored(sharpened_image, None, 10, 10, 7, 15)

    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(test, kernel, iterations=1)
    test2 = cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)
    test3 = cv2.threshold(test2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    test4 = 255 - test3
    karnel2 = np.ones((3, 3), np.uint8)
    er = cv2.erode(test4, karnel2, iterations=1)
    config = '--psm 7'
    text = pytesseract.image_to_string(er, config=config)
    text_words = text.split(' ')

    card_title = ' '.join([str(elem) for elem in text_words])
    return card_title


def main():
    img_path = "images/img_9.png"
    image = cv2.imread(img_path)
    cards = detect_cards(image)
    cards_titles_areas = []
    for card in cards:
        card_title = get_card_title_area(card)
        if card_title is not None:
            cards_titles_areas.append(card_title)

    cards_titles = []
    for card_title_area in cards_titles_areas:
        card_title = get_card_title(card_title_area)
        cards_titles.append(card_title)

    for cards_title in cards_titles:
        characters_to_remove = ['{', '}', '(', ')', '|', ':', ']',
                                '[', '\n', '\t', '*', '_', '%', '#',
                                '@', '!', '?', '/', ';', '<', '>', '^', '$', '&', '*', '~', '`']
        rx = '[' + re.escape(''.join(characters_to_remove)) + ']'
        result = re.sub(rx, '', cards_title)
        print(result.strip())


if __name__ == "__main__":
    main()
