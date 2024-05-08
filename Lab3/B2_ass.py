# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 00:24:24 2024

@author: Asus
"""

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def double_gaussian(sigma1, mean1, sigma2, mean2):
    out = np.zeros(256, dtype=np.float32)
    c1 = (sigma1 * np.sqrt(2 * np.pi))
    c2 = (sigma2 * np.sqrt(2 * np.pi))
    sig1 = 2 * sigma1 ** 2
    sig2 = 2 * sigma2 ** 2
    for i in range(256):
        p_x = np.exp(-((i - mean1) ** 2) / (sig1)) / (c1) + np.exp(-((i - mean2) ** 2) / (sig2)) / (c2)
        out[i] = p_x

    return out

def histogram_equalization(image):
    histogram = np.zeros(256, dtype=np.uint32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image[i, j]
            histogram[pixel] += 1

    pdf = np.zeros(256, dtype=np.float32)
    m, n = image.shape
    size = m * n
    for i in range(0, 256):
        pdf[i] = histogram[i] / size

    cdf = np.zeros(256, dtype=np.float32)
    cdf[0] = pdf[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + pdf[i]

    for i in range(0, 256):
        cdf[i] = round(cdf[i] * 255)

    equalized_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            equalized_image[i, j] = cdf[image[i, j]]

    plt.title('Equalized Histogram')
    return pdf, cdf, equalized_image


image = cv2.imread('histogram.jpg', cv2.IMREAD_GRAYSCALE)
# #
# mean1 = input("Enter mean1: ")
# simga1 = input("Enter sigma1: ")
# mean2 = input("Enter mean2: ")
# sigma2 = input("Enter sigma2: ")
pdf, cdf, equalized_image = histogram_equalization(image)
# g_distribution = double_gaussian(simga1, mean1, sigma2, mean2)

g_distribution = double_gaussian(8, 30, 20, 160)
# g_distribution = double_gaussian(20, 50, 30, 100)


g_pdf = g_distribution / np.sum(g_distribution)


g_cdf = np.zeros(256)
g_cdf[0] = g_pdf[0]
for i in range(1, 256):
    g_cdf[i] = g_cdf[i - 1] + g_pdf[i]

for i in range(0, 256):
    g_cdf[i] = round(g_cdf[i] * 255)

plt.subplot(2, 2, 1)
plt.title("target pdf")
plt.plot(g_pdf)

plt.subplot(2, 2, 2)
plt.title("target cdf")
plt.plot(g_cdf)

plt.subplot(2, 2, 3)
plt.title("equalized image histogram")
plt.hist(equalized_image.ravel(), 256, [0, 255])

plt.show()


map = np.zeros(256)
for i in range(256):
    x = (np.abs(g_cdf - cdf[i])).argmin()
    map[i] = x

output = np.zeros_like(image)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        output[i, j] = map[image[i, j]]

o_pdf, o_cdf, o_equalized_image = histogram_equalization(output)

plt.subplot(3, 2, 1)
plt.title("input histogram")
plt.hist(image.ravel(), 256, [0, 255])

plt.subplot(3, 2, 2)
plt.title("pdf")
plt.plot(pdf)

plt.subplot(3, 2, 3)
plt.title("cdf")
plt.plot(cdf)

plt.subplot(3, 2, 4)
plt.title("matched histogram")
plt.hist(output.ravel(), 256, [0, 255])

plt.subplot(3, 2, 5)
plt.title("pdf")
plt.plot(o_pdf)

plt.subplot(3, 2, 6)
plt.title("cdf")
plt.plot(o_cdf)
plt.show()

cv2.imshow("input image", image)
cv2.imshow("equalized image", equalized_image)
cv2.imshow("matched image", output)

cv2.waitKey(0)
cv2.destroyAllWindows()