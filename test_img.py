import cv2
import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage import io
from skimage import segmentation


def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	#m = mse(imageA, imageB)
	s = ssim(imageA, imageB,multichannel=True)
	#print(m)
	print(s)

image = cv2.imread("img1.jpg")
image2 = cv2.imread("img2.jpg")
#height, width = image2.shape[:2]
print(image.dtype)
print(image2.dtype)
compare_images(image,image2,"Maaki Choot")