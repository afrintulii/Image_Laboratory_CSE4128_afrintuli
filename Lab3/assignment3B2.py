# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 00:21:47 2024

@author: Asus
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def double_gaussian(sigma1,mean1,sigma2,mean2):
    g_out = np.zeros(256,dtype=np.float32)
    c1=(sigma1*np.sqrt(2*np.pi))
    c2=(sigma2*np.sqrt(2*np.pi))
    
    sig1= 2 * sigma1 **2 
    sig2= 2 * sigma2 **2
    for i in range(256):
        sum = np.exp(-((i-mean1)**2)/(sig1)) / (c1) + np.exp(-((i-mean2) **2) /(sig2)) /(c2)
        g_out[i] = sum
    return g_out
def double_gaussian1(sigma1, mean1, sigma2, mean2):
    out = np.zeros(256, dtype=np.float32)
    c1 = (sigma1 * np.sqrt(2 * np.pi))
    c2 = (sigma2 * np.sqrt(2 * np.pi))
    sig1 = 2 * sigma1 ** 2
    sig2 = 2 * sigma2 ** 2
    for i in range(256):
        p_x = np.exp(-((i - mean1) ** 2) / (sig1)) / (c1) + np.exp(-((i - mean2) ** 2) / (sig2)) / (c2)
        out[i] = p_x

    return out
def histogram_equalization(img):
    histogram = np.zeros(256,dtype=np.float32)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = img[i,j]
            histogram[pixel]+=1
    pdf = np.zeros(256,dtype=np.float32)
    size = img.shape[0]*img.shape[1]
    for i in range(0,256):
        pdf[i] = histogram[i]/size
    cdf = np.zeros(256,dtype=np.float32)
    cdf[0] = pdf[0]
    new_hist =np.zeros(256,dtype=np.float32)
    for i in range(1,256):
        cdf[i] = pdf[i] + cdf[i-1]
       
    rcdf=np.zeros(256,dtype=np.uint32)    
    for i in range(0,256):
        rcdf[i] = round(cdf[i]*255)
        new_hist[rcdf[i]] = histogram[i]
    
    equalized_image = np.zeros_like(img)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            equalized_image[i,j] = rcdf [img[i,j]]
    
    return pdf,rcdf,equalized_image,new_hist
    

img=cv2.imread('histogram.jpg',cv2.IMREAD_GRAYSCALE)

pdf,cdf,equalized_image,mp_func = histogram_equalization(img)

g_distribution = double_gaussian (8, 30, 20, 165)

g_pdf = g_distribution / np.sum(g_distribution)

g_cdf = np.zeros(256,dtype=np.float32)

g_cdf[0]= g_pdf[0]

for i in range(1,256):
    g_cdf[i] =g_pdf[i] + g_cdf[i-1]
rg_cdf=np.zeros(256,dtype=np.uint32)    
for  i in range(0,256):
    g_cdf[i] = round(g_cdf[i]*255)

plt.subplot(2,2,1)
plt.title("target pdf")
plt.plot(g_pdf)

plt.subplot(2,2,2)
plt.title("target cdf")
plt.plot(g_cdf)
plt.show()

mp = np.zeros(256,dtype = np.float32)
"""output_img = np.zeros_like(img)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        output_img[i,j] = mp_func[rg_cdf[img[i,j]]]
cv2.imshow("output_img",output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()"""
for i in range(256):
    x = (np.abs(g_cdf-cdf[i])).argmin()
    mp[i] = x
output=np.zeros_like(img)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        output[i,j] = mp[img[i,j]]

o_pdf,o_cdf,o_equalized_image,o_mp_func = histogram_equalization(output)

plt.subplot(3,2,1)
plt.title("input histogram")
plt.hist(img.ravel(),256,[0,255])

plt.subplot(3,2,2)
plt.title("pdf of input image")
plt.plot(pdf)

plt.subplot(3,2,3)
plt.title("cdf of input image")
plt.plot(cdf)

plt.subplot(3,2,4)
plt.title("output histogram")
plt.hist(output.ravel(),256,[0,255])

plt.subplot(3,2,5)
plt.title("pdf of output image")
plt.plot(o_pdf)

plt.subplot(3,2,6)
plt.title("cdf of output image")
plt.plot(o_cdf)

plt.show()



cv2.imshow("input_img",img)
cv2.imshow("Equalized_img",equalized_image)
cv2.imshow("output_img",output)
cv2.imshow("output_equalized_img",o_equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()