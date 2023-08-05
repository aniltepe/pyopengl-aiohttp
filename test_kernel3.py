import numpy as np
import cv2 as cv
from scipy.signal import convolve2d
from skimage.transform import rescale
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte, img_as_float


test_img = cv.imread('image.png')
kernel = np.array([[-1, -1, -1], 
                   [-1, 8, -1],
                   [-1, -1, -1]])
# gray_test = rgb2gray(img_as_float(test_img))
gray_test = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)
gray_test = cv.equalizeHist(gray_test)
conv_test = abs(convolve2d(img_as_float(gray_test), kernel).clip(0,1))
conv_test = img_as_ubyte(conv_test)
cv.imshow("test", conv_test)
cv.waitKey(0)