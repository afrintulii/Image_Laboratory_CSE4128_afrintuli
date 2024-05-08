# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:35:13 2024

@author: Nihal
"""

import numpy as np
import cv2

def applyFilter(img, kernel, cx = None, cy = None):
    
    if cx == None:
        cx = kernel.shape[0] // 2
    if cy == None:
        cy = kernel.shape[1] // 2
            
    kleft = -(cx - 0)
    kright = (kernel.shape[0] - 1) - cx 
    ktop = -(cy - 0)
    kbottom = (kernel.shape[1] - 1) - cy
    
    bordered_img = cv2.copyMakeBorder(src = img, top = -ktop, bottom = kbottom, left = -kleft, right = kright, borderType = cv2.BORDER_CONSTANT)
    out = np.zeros((img.shape[0], img.shape[1]))
    
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            for x in range(ktop, kbottom + 1):
                for y in range(kleft, kright + 1):
                    out[i, j] += bordered_img[(i - ktop) + x, (j - kleft) + y] * kernel[(kernel.shape[0] - 1) - (x - ktop), (kernel.shape[1] - 1) - (y - kleft)]
    
    return out

def applyZeroCrossing(img):
    
    out = np.zeros((img.shape[0], img.shape[1]))
    
    for i in range(0 + 1, img.shape[0] - 1):
        for j in range(0 + 1, img.shape[1] - 1):
            if np.sign(img[i, j-1]) != np.sign(img[i, j+1]):
                out[i, j] = img[i, j]
            if np.sign(img[i-1, j]) != np.sign(img[i+1, j]):
                out[i, j] = img[i, j]
    
    return out

def applyThresholding(img, zero):
    
    for i in range(zero.shape[0]):
        for j in range(zero.shape[1]):
            localRegion = img[max(0, i-1) : min(zero.shape[0]-1, i+1), max(0, j-1) : min(zero.shape[1]-1, j+1)]
            stdDev = np.std(localRegion)
            if (zero[i, j] >= stdDev):
                zero[i, j] = 255
            else:
                zero[i, j] = 0
    
    return zero

def genLoGFilter(s):
    
    kx = (int)(max(7, 7 * s))
    ky = (int)(max(7, 7 * s))
    
    kernel = np.zeros((kx, ky))
    
    cx = kx // 2
    cy = ky // 2
    
    kleft = -(cx - 0)
    kright = (kernel.shape[0] - 1) - cx 
    ktop = -(cy - 0)
    kbottom = (kernel.shape[1] - 1) - cy
    
    for x in range(ktop, kbottom + 1):
        for y in range(kleft, kright + 1):
            term = (x**2 + y**2) / (2 * s**2)
            kernel[x - ktop, y - kleft] = - (1 / np.pi * s**4) * (1 - term) * np.exp(-term)
            
    return kernel
    

img = cv2.imread('Lena.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow("Grayscale", img)
cv2.waitKey(0)

s = 1

loGKernel = genLoGFilter(s)

outLoG = applyFilter(img, loGKernel)
outLoG2 = outLoG.copy()

cv2.normalize(src = outLoG2, dst = outLoG2, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
outLoG2 = np.round(outLoG2).astype(dtype = np.uint8)

cv2.imshow("outLog",outLoG)
cv2.waitKey(0)
cv2.imshow("LoG User-defined", outLoG2)
cv2.waitKey(0)

outZero = applyZeroCrossing(outLoG)

cv2.imshow("Zero Crossing", outZero)
cv2.waitKey(0)

threshold = applyThresholding(img, outZero)

cv2.imshow("Threshold", threshold)
cv2.waitKey(0)

cv2.destroyAllWindows()