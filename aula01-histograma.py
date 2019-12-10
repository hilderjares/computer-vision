import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

filename = sys.argv[1]
image = cv2.imread(filename, 0)

print (image)
print (image.shape)

width = image.shape[1]
height = image.shape[0]

#ler todos os pixels da imagem
for c in range(0, width - 1):
	for l in range(0, height - 1):
		pixel = image.item(l, c)

		#negativo
		image.itemset(l , c, 255 - pixel)

#		if px > 100:
#			im.itemset(l,c,255)

#redimensiona a imagem
def resizeImage(image):
	new_width = int(image.shape[1] * .5)
	new_height = int(image.shape[0] * .5)
	dim = (new_width, new_height)
	im_resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

#mostra imagem com opencv
def showImageWithOpencv(image):
	#cv2.imshow('resized',im_resized)
	cv2.imshow('imagem', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#calcula o histograma
def calculateHistograma(image, bins = 256):
	histograma = cv2.calcHist([image], [0], None, [bins], [0, bins])

	return histograma
	
#mostra o histograma
def showHistograma(histograma, bins = 256):
	x_coord = np.arange(bins)
	# plt.bar(x_coord, hist)
	# plt.xlim([0, bins])
	# plt.show()
	return x_coord

#Equaliza histograma com CDF
def equalizeHistogramaWithCdf(image):	
	im_eq_cdf = cv2.equalizeHist(image)
	cv2.imshow('im-eq', im_eq_cdf)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#Equaliza histograma com ajuste local e constrate limitado
def equalizeHistogramaWithClahe(image):
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	im_eq_clahe = clahe.apply(image)
	cv2.imshow('imagem', image)
	cv2.imshow('image-clahe', im_eq_clahe)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#mostra imagem com matplotlib
def showImageWithMatplotlib(image, hist, x_coord):	
	plt.subplot(221), plt.imshow(image, cmap='gray')
	plt.subplot(222), plt.bar(x_coord, hist)
	plt.subplot(223), plt.imshow(im_eq_clahe, cmap='gray')
	plt.subplot(224), plt.hist(im_eq_clahe.ravel(), 256, [0, 256])
	plt.xlim([0,256])
	plt.show()

if __name__ == "__main__":
	hist = calculateHistograma(image, 256)
	x_coord = showHistograma(hist, 256)
	# calculateHistograma(image)
	# showHistograma()
	# showImageWithOpencv(image)
	equalizeHistogramaWithCdf(image)
	# equalizeHistogramaWithClahe(image)
	# showImageWithMatplotlib(image, hist, x_coord)
