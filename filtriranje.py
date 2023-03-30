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
    magnitude = np.sqrt(np.power(gradient_x, 2) + np.power(gradient_y, 2)).astype(np.uint8)
    return magnitude


def my_prewitt(image):
    prewitt_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    prewitt_kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    gradient_x = convolve(image, prewitt_kernel_x)
    gradient_y = convolve(image, prewitt_kernel_y)
    output=np.sqrt(np.power(gradient_x, 2) + np.power(gradient_y, 2)).astype(np.uint8)
    return output



def my_sobel(image):
    
    return slika_robov


def canny(slika, sp_prag, zg_prag):
    #vaša implementacija
    return slika_robov 

def spremeni_kontrast(slika, alfa, beta):
    pass


imgGray = cv2.imread('lenna.png',0)
cv2.namedWindow("Slika")
cv2.imshow("Slika", imgGray)

roberts = my_roberts(imgGray)
cv2.namedWindow("my_roberts")
cv2.imshow("my_roberts", roberts)

prewitt = my_roberts(imgGray)
cv2.namedWindow("my_prewitt")
cv2.imshow("my_prewitt", prewitt)



cv2.waitKey()
cv2.destroyAllWindows()