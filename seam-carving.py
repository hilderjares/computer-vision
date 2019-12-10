import cv2 as cv
import numpy as np
from skimage import transform
from skimage import filters
import sys

filename = sys.argv[1]

image = cv.imread(filename)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

mag = filters.sobel(gray.astype("float"))

# show the original image
cv.imshow("Original", image)

for numSeams in range(100, 240, 100):
    # vertical or horizontal
    carved = transform.seam_carve(image, mag, 'vertical', numSeams)

    print("[INFO] removing {} seams; new size: "
          "w={}, h={}".format(numSeams, carved.shape[1],
                              carved.shape[0]))

    cv.imshow("Carved", carved)
    cv.waitKey(0)
