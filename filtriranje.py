import cv2
import numpy as np

def convolve(image, kernel):
    # Get the dimensions of the image and kernel
    img_height, img_width = image.shape[:2]
    kernel_height, kernel_width = kernel.shape[:2]

    # Compute the padding required around the input image
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Pad the input image with zeros
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), 'constant')

    # Compute the output image
    output = np.zeros_like(image)
    for y in range(img_height):
        for x in range(img_width):
            patch = padded_image[y:y + kernel_height, x:x + kernel_width]
            output[y, x] = np.sum(patch * kernel)

    return output

def my_roberts(image):
    roberts_kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    gradient_x = convolve(image, roberts_kernel_x)
    gradient_y = convolve(image, roberts_kernel_y)
    output = np.sqrt(np.power(gradient_x, 2) + np.power(gradient_y, 2)).astype(np.uint8)
    return output

def my_prewitt(image):
    prewitt_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    prewitt_kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    gradient_x = convolve(image, prewitt_kernel_x)
    gradient_y = convolve(image, prewitt_kernel_y)
    output=np.sqrt(np.power(gradient_x, 2) + np.power(gradient_y, 2)).astype(np.uint8)
    return output



def my_sobel(image):
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gradient_x = convolve(image, sobel_kernel_x)
    gradient_y = convolve(image, sobel_kernel_y)
    output=np.sqrt(np.power(gradient_x, 2) + np.power(gradient_y, 2)).astype(np.uint8)
    
    return output



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



imgGray2 = cv2.imread('lenna.png',0)
cv2.namedWindow("Slika2")
cv2.imshow("Slika2", imgGray2)

gauss = cv2.GaussianBlur(imgGray2,(15,15),1)
imgGray = cv2.hconcat((imgGray2,gauss))

#imgGray=spremeni_kontrast(imgGray2,1.2,-50)
cv2.namedWindow("Slika")
cv2.imshow("Slika", imgGray)


roberts = my_roberts(imgGray)
cv2.namedWindow("my_roberts")
cv2.imshow("my_roberts", roberts)

prewitt = my_prewitt(imgGray)
cv2.namedWindow("my_prewitt")
cv2.imshow("my_prewitt", prewitt)

sobel = my_sobel(imgGray)
cv2.namedWindow("my_sobel")
cv2.imshow("my_sobel", sobel)

can = canny(imgGray,100,200)
cv2.namedWindow("canny")
cv2.imshow("canny", can)

cv2.waitKey()
cv2.destroyAllWindows()