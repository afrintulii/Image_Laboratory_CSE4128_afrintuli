# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 01:25:07 2024

@author: Asus
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def histrogram_equlization(image):
    histrogram= np.zeros(256,dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel=image[i,j]
            histrogram[pixel]+=1
            
    plt.figure(1)
    plt.title("input Histogram")
    plt.plot(histrogram)
    plt.show()
    
    pdf = np.zeros(256,dtype=np.float32)
    m,n=image.shape
    size = m*n
    for i in range(0,256):
        pdf[i] = histrogram[i]/size
    
    plt.figure(2)
    plt.title("pdf")
    plt.plot(pdf)
    plt.show()
    """cdf = np.zeros(256, dtype=np.float32)
    cdf[0] = pdf[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + pdf[i]

    for i in range(1, 256):
        cdf[i] = round(cdf[i] * 255)

    plt.figure(3)
    plt.title("CDF")
    plt.plot(cdf)
    plt.show()"""
    cdf = np.zeros(256,dtype = np.float32)
    cdf[0] =pdf[0]
    for i in range(1,256):
        cdf[i] = cdf[i-1] + pdf[i]
    
    for i in range(1,256):
        cdf[i] = round(cdf[i]*255)
        
    plt.figure(3)
    plt.title("CDF")
    plt.plot(cdf)
    plt.show()
    equalized_image = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            equalized_image[i,j]=cdf[image[i,j]]
    cv2.imshow("original image",image)
    cv2.imshow("histrogram image",equalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    

image = cv2.imread('histogram.jpg',cv2.IMREAD_GRAYSCALE)
equlized_image= histrogram_equlization(image)