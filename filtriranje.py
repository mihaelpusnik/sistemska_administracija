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
    
    return slika_robov


def my_prewitt(slika):
    #vaša implementacija
    return slika_robov 

def my_sobel(slika):
    #vaša implementacija
    return slika_robov 

def canny(slika, sp_prag, zg_prag):
    #vaša implementacija
    return slika_robov 

def spremeni_kontrast(slika, alfa, beta):
    pass


imgGray = cv2.imread('lenna.png',0)
cv2.namedWindow("Slika")
cv2.imshow("Slika", imgGray)
img = my_roberts(imgGray)
cv2.namedWindow("Slika2")
cv2.imshow("Slika2", img)
cv2.waitKey()
cv2.destroyAllWindows()