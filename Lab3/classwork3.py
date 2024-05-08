import cv2
import numpy as np
import matplotlib.pyplot as plt


def histogram_equalization(image):
    # histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

    histogram = np.zeros(256, dtype=np.uint32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image[i, j]
            histogram[pixel] += 1

    plt.figure(1)
    plt.title("input Histogram")
    plt.plot(histogram)
    plt.show()

    pdf = np.zeros(256, dtype=np.float32)
    m, n = image.shape
    size = m * n
    for i in range(0, 256):
        pdf[i] = histogram[i] / size

    plt.figure(2)
    plt.title("PDF")
    plt.plot(pdf)
    plt.show()

    cdf = np.zeros(256, dtype=np.float32)
    cdf[0] = pdf[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + pdf[i]

    for i in range(1, 256):
        cdf[i] = round(cdf[i] * 255)

    plt.figure(3)
    plt.title("CDF")
    plt.plot(cdf)
    plt.show()

    equalized_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            equalized_image[i, j] = cdf[image[i, j]]

    cv2.imshow('Original Image', image)
    cv2.imshow('Equalized Image', equalized_image)

    # equalized_image = np.uint8(equalized_image)
    # histogram = cv2.calcHist(equalized_image, [0], None, [256], [0, 256])
    histogram = np.zeros(256, dtype=np.uint32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = equalized_image[i, j]
            histogram[pixel] += 1

    plt.figure(4)
    plt.title("output histogram")
    plt.plot(histogram)
    plt.show()



    pdf = np.zeros(256, dtype=np.float32)
    m, n = image.shape
    size = m * n
    for i in range(0, 256):
        pdf[i] = histogram[i] / size

    plt.figure(5)
    plt.title("PDF")
    plt.plot(pdf)
    plt.show()

    cdf = np.zeros(256, dtype=np.float32)
    cdf[0] = pdf[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + pdf[i]

    for i in range(1, 256):
        cdf[i] = round(cdf[i] * 255)

    plt.figure(6)
    plt.title("CDF")
    plt.plot(cdf)
    plt.show()

    return equalized_image


image = cv2.imread('histogram.jpg', cv2.IMREAD_GRAYSCALE)

equalized_image = histogram_equalization(image)

cv2.waitKey(0)
cv2.destroyAllWindows()
