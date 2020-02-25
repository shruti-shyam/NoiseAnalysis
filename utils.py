import scipy
from scipy.ndimage import median_filter
import numpy as np
import cv2
import os


def ELA(image, quality, amplification, flatten=True):
    """
    :param image: input image
    :param quality: JPEG quality factor between [0,100]
    :param amplification: amplification factor to magnify difference
    :param flatten: set to True if returning result in gray scale
    :return: magnified error between original and JPEG compressed image
    """
    cv2.imwrite("compressed_img.jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    compressed_image = cv2.imread("compressed_img.jpg")
    ela = (np.asarray(image) - np.asarray(compressed_image)) * amplification
    if flatten:
        ela = np.mean(ela, axis=2)
    return ela


def median_filter_analysis(image, filter_size=5):
    # apply histogram equalisation on input image
    # cv2.equalizeHist() works only on grayscale image
    # Assumption: noise artifacts should be visible in grayscale too
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalised_image = cv2.equalizeHist(gray_image)
    filtered_image = median_filter(equalised_image, size=filter_size)
    noise_image = equalised_image - filtered_image
    return noise_image


def get_patch_statistics(img_patch):
    # accepts only gray scale images
    # return mean and variance of patch
    img = np.asarray(img_patch)
    return np.mean(img), np.var(img)
