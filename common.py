import cv2
import numpy as np


def resize_image(img, new_height):
    """Resizes source image to provided max height maintaining aspect ratio"""
    (height, width) = img.shape[:2]
    ratio = new_height / float(height)
    dim = (int(width * ratio), new_height)
    return (cv2.resize(img, dim, interpolation=cv2.INTER_AREA), ratio)


def get_edge_detection_thresholds(img, sigma_value):
    """Calculates the lower and upper thresholds for Canny edge detection"""
    sigma = sigma_value
    median = np.median(img)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return (lower, upper)