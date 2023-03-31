import cv2
import numpy as np


def my_roberts(slika):

	# Define Roberts kernels
	roberts_x = np.array([[0, 1], [-1, 0]])
	roberts_y = np.array([[1, 0], [0, -1]])
	h,w = slika.shape
	# Compute gradient images
	slika_robov=np.zeros((h,w))
	# Compute magnitude of gradients
	for i in range(0, h - 1):
		for j in range(0, w - 1):
			# Compute gradient in the y direction
			gy= (roberts_y[0][0] * slika[i][j]) + (roberts_y[0][1] * slika[i][j + 1]) + \
				(roberts_y[1][0] * slika[i + 1][j]) + (roberts_y[1][1] * slika[i + 1][j + 1])

			# Compute gradient in the x direction
			gx = (roberts_x[0][0] * slika[i][j]) + \
				(roberts_x[0][1] * slika[i][j + 1]) + \
				(roberts_x[1][0] * slika[i + 1][j]) + \
				(roberts_x[1][1] * slika[i + 1][j + 1])

			slika_robov[i][j] = np.sqrt(gx**2 + gy**2)

	# Normalize magnitude image to range [0, 255]
	slika_robov = cv2.normalize(slika_robov, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

	return slika_robov


def my_prewitt(image):
	# Define Prewitt kernels
	prewitt_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
	prewitt_kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

	h, w = image.shape
	slika_robov=np.zeros((h,w))

	for i in range(1, h - 1):
		for j in range(1, w - 1):
			# Compute gradient in the x direction
			gradient_x = np.sum(image[i-1:i+2, j-1:j+2] * prewitt_kernel_x)
			# Compute gradient in the y direction
			gradient_y= np.sum(image[i-1:i+2, j-1:j+2] * prewitt_kernel_y)
			slika_robov[i][j] = np.sqrt(gradient_x**2 + gradient_y**2)

	# Compute magnitude of gradients

	# Normalize magnitude image to range [0, 255]
	slika_robov = cv2.normalize(slika_robov, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

	return slika_robov




def my_sobel(slika):

	# Define Sobel kernels
	sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
	sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
	h,w = slika.shape
	# Compute gradient images
	slika_robov=np.zeros((h,w))
	# Compute magnitude of gradients
	for i in range(1, h - 1):
		for j in range(1, w - 1):
			# Compute gradient in the y direction
			gy= (sobel_y[0][0] * slika[i - 1][j - 1]) + (sobel_y[0][1] * slika[i - 1][j]) + \
				(sobel_y[0][2] * slika[i - 1][j + 1]) + (sobel_y[1][0] * slika[i][j - 1]) + \
				(sobel_y[1][1] * slika[i][j]) + (sobel_y[1][2] * slika[i][j + 1]) + \
				(sobel_y[2][0] * slika[i + 1][j - 1]) + (sobel_y[2][1] * slika[i + 1][j]) + \
				(sobel_y[2][2] * slika[i + 1][j + 1])

			# Compute gradient in the x direction
			gx = (sobel_x[0][0] * slika[i - 1][j - 1]) + \
				(sobel_x[0][1] * slika[i - 1][j]) + \
				(sobel_x[0][2] * slika[i - 1][j + 1]) + \
				(sobel_x[1][0] * slika[i][j - 1]) + \
				(sobel_x[1][1] * slika[i][j]) + \
				(sobel_x[1][2] * slika[i][j + 1]) + \
				(sobel_x[2][0] * slika[i + 1][j - 1]) + \
				(sobel_x[2][1] * slika[i + 1][j]) + \
				(sobel_x[2][2] * slika[i + 1][j + 1])

			slika_robov[i][j] = np.sqrt(gx**2 + gy**2)

	# Normalize magnitude image to range [0, 255]
	slika_robov = cv2.normalize(slika_robov, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

	return slika_robov



def canny(image, lower_threshold, higher_threshold):
    edges = cv2.Canny(image, lower_threshold, higher_threshold)
    return edges

#alfa je nastavljena med 0,5 in 2, beta pa med -100 in 100.
def spremeni_kontrast(image, alpha, beta):
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def spremeni_kontrast2(image, alpha, beta):
    adjusted = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)
    return adjusted

def smooth(image, kernel_size=5, sigma=1):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)



imgGray = cv2.imread('lenna.png',0)
cv2.namedWindow("Slika2")
cv2.imshow("Slika2", imgGray)

#gauss = cv2.GaussianBlur(imgGray2,(15,15),1)
#imgGray = cv2.hconcat((imgGray2,gauss))

#imgGray=spremeni_kontrast(imgGray2,1.2,-50)
#cv2.namedWindow("Slika")
#cv2.imshow("Slika", imgGray)


roberts = my_roberts(imgGray)
cv2.namedWindow("my_roberts")
cv2.imshow("my_roberts", roberts)

prewitt = my_prewitt(imgGray)
cv2.namedWindow("my_prewitt")
cv2.imshow("my_prewitt", prewitt)

sobel = my_sobel(imgGray)
cv2.namedWindow("my_sobel")
cv2.imshow("my_sobel", sobel)

can = canny(imgGray,10,100)
cv2.namedWindow("canny")
cv2.imshow("canny", can)

cv2.waitKey()
cv2.destroyAllWindows()