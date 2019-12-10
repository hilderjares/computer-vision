import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def gamma_correction(img, gamma, c=1.0):
    i = img.copy()
    i[:, :, :] = 255 * (c * (img[:, :, :] / 255.0)**(1.0 / gamma))
    return i


def gamma_correction_LUT(img, gamma, c=1.0):

    transf_table = np.array([
        c * ((i / 255.0)**(1.0 / gamma)) * 255 for i in np.arange(0, 256)
    ]).astype("uint8")

    # transform using the lookup table
    return cv2.LUT(img, transf_table)


#abre imagem
filename = sys.argv[1]
im = cv2.imread(filename)

#converte cores
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

#split
#im_r,im_g,im_b = cv2.split(im)
im_r = im[:, :, 0]
im_g = im[:, :, 1]
im_b = im[:, :, 2]

#calcula o histograma de cada canal
bins = 256
hist_r = cv2.calcHist([im_r], [0], None, [bins], [0, bins])
hist_g = cv2.calcHist([im_g], [0], None, [bins], [0, bins])
hist_b = cv2.calcHist([im_b], [0], None, [bins], [0, bins])

print (im_r.mean())
print (im_g.mean())
print (im_b.mean())

#combina as imagens
im = cv2.merge([im_r, im_g, im_b])

#mostra imagens
imagens = [im, im_r, im_g, im_b]
titles = ['original', 'Red', 'Green', 'Blue']
for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(imagens[i], cmap='gray')
    plt.title(titles[i]), plt.xticks([]), plt.yticks([])

plt.show()

#Correcao gamma
im_gamma = gamma_correction(im, 5.0)

imagens = [im, im_gamma]
titles = ['original', 'gamma']
for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(imagens[i], cmap='gray')
    plt.title(titles[i]), plt.xticks([]), plt.yticks([])

plt.show()
