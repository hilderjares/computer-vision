import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

# img = cv2.imread('./imagens/blog-03.10.jpg')

filename = sys.argv[1]

img = cv2.imread(filename)

#media 5x5
avg_blur = cv2.blur(img,(5,5))

#gaussian
gaussian_blur = cv2.GaussianBlur(img,(5,5),1)

#mediana
median_blur = cv2.medianBlur(img,5)

#bilateral
bilateral_blur = cv2.bilateralFilter(img,9,100,100)

#custom kernel (media 3x3)
kernel = np.float32([[1,1,1],[1,1,1],[1,1,1]])/9
custom = cv2.filter2D(img,-1,kernel)


plt.subplot(231),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(avg_blur),plt.title('Media')
plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(gaussian_blur),plt.title('gaussian')
plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(median_blur),plt.title('mediana')
plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(bilateral_blur),plt.title('bilateral')
plt.xticks([]), plt.yticks([])
plt.subplot(236),plt.imshow(custom),plt.title('custom')
plt.xticks([]), plt.yticks([])

plt.show()