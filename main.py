import cv2
import numpy as np
import imutils
from PIL import Image
import imagehash
import pytesseract
from itertools import combinations

image = cv2.imread('img_7.png')


def find_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(image=gray, threshold1=100, threshold2=200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    max_area = max(cnts, key = cv2.contourArea)
    max_area = cv2.contourArea(max_area)
    screenCnts = []
    for c in cnts:
        area = cv2.contourArea(c)
        if max_area * 0.8 <= area <= max_area * 1.2:
            screenCnt = c
            screenCnts.append(screenCnt)
    return screenCnts


def get_opposites(cnt):
    current_max = 0
    c = None
    for a, b in combinations(cnt, 2):
        current_distance = np.linalg.norm(a - b)
        if current_distance > current_max:
           current_max = current_distance
           c = a, b
    return c


def calculate_hash(image):
    image = Image.fromarray(image)
    return imagehash.average_hash(image)


def similar_hashes(hash_to_check, hashes, cutoff=8):
    # Compare each image to every other image
    new_hashes = []
    for hash in hashes:
        if hash_to_check - hash < cutoff:
            return True


def main():
    cnts = find_edges(image)
    saved_hashes = set()
    for i, cnt in enumerate(cnts):
        xs = cnt[..., 0]
        ys = cnt[..., 1]
        x_mid = (np.amin(xs) + np.amax(xs)) // 2
        y_mid = (np.amin(ys) + np.amax(ys)) // 2
        tl_br = cnt[((ys < y_mid) & (xs < x_mid)) | ((ys > y_mid) & (xs > x_mid))]
        tr_bl = cnt[((ys > y_mid) & (xs < x_mid)) | ((ys < y_mid) & (xs > x_mid))]

        p1, p3 = get_opposites(tl_br)
        p2, p4 = get_opposites(tr_bl)
        cv2.polylines(image, np.array([[p1, p2, p3, p4]], np.int32), True, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / h
        if 0.4 <= ratio <= 0.9:
            cropped = image[y:y + h, x:x + w]
            cropped_hash = calculate_hash(cropped)
            if (cropped_hash not in saved_hashes) and not similar_hashes(cropped_hash, saved_hashes):
                cv2.imwrite(f'cropped_{i}.jpg', cropped)
                roi = cropped[0:80, 0::]
                gray2 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                cv2.imwrite('roi.png', gray2)
                data = pytesseract.image_to_data(gray2, output_type=pytesseract.Output.DICT)

                words_vote = list(zip(data['text'], data['conf']))
                # Filter the list of recognized words
                words = [word[0] for word in words_vote if int(word[1]) > 50]
                final_string = ' '.join(words).strip()
                print(final_string)
                saved_hashes.add(cropped_hash)

if __name__ == "__main__":
    main()