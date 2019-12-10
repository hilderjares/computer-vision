import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

filename = sys.argv[1]

img = cv2.imread(filename, 0)

rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)  # center

# fft img
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 1], dft_shift[:, :, 1]))
 
mask = dft_shift.copy()
mask[:, :] = 0

img_mask_convert = mask

cv2.circle(img_mask_convert, (ccol, crow), 63, (1.0,1.0), -1)

# apply mask and inverse DFT
fshift = cv2.multiply(dft_shift, img_mask_convert)

mask = cv2.magnitude(mask[:, :, 0], mask[:, :, 1])

fshift_mask_mag = np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]) + 1)

f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(mask, cmap='gray')
plt.title('After FFT'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(fshift_mask_mag, cmap='gray')
plt.title('FFT + Mask'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(img_back, cmap='gray')
plt.title('After FFT Inverse'), plt.xticks([]), plt.yticks([])

plt.show()
