import cv2
import numpy as np
import pytesseract


def main():
    pytesseract.pytesseract.tesseract_cmd = r'C:\Programy\Tesseract-OCR\tesseract.exe'

    # Load the image
    image = cv2.imread('cards_img.png')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold
    _, threshold = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY)

    # Use Tesseract to read the text
    text = pytesseract.image_to_string(threshold)

    # Print the text
    print(text)


if __name__ == "__main__":
    main()