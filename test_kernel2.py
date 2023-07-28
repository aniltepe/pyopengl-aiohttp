import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.transform import rescale
from scipy.signal import convolve2d

my_dog = imread('figure.png')
my_dog_gray = rgb2gray(my_dog[:,:,:3])
kernel = np.array([[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]])
conv_im1 = convolve2d(my_dog_gray, kernel[::-1, ::-1]).clip(0,1)
plt.imshow(abs(conv_im1), cmap='gray')
# plt.imshow(my_dog)
plt.show()