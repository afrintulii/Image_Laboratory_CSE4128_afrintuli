# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 21:45:10 2024

@author: Asus
"""

import numpy as np
import math
import cv2

def get_log_kernel(sigma):
    kx = int(max(7,7**sigma))
    ky = int(max(7,7*sigma))
    
    kernel = np.zeros((kx,ky))
    
    cx = int(kx//2 )
    cy = int(ky//2)
    
    kleft = int(-(cx-0))
    kright = int((kernel.shape[0]-1) - cx)
    ktop = - int(cy-0)
    kbottom = int((kernel.shape[1]-1) - cy)
    
    for x in range(ktop,kbottom+1):
        for y in range(kleft,kright+1):
            term = ((x**2) + (y**2)) / (2*(sigma**2))
            kernel[x-ktop,y-kleft] = -(1/(np.pi* sigma**4 )) * (1-term) * np.exp(-term)
    return kernel

def Convolution(img,kernel,cx=None,cy=None):
    
    if cx == None:
        cx = kernel.shape[0]//2
    if cy == None:
        cy = kernel.shape[1]//2
    
    kleft = -int(cx-0)
    kright = int((kernel.shape[0] -1) -cx)
    ktop = -int(cy-0)
    kbottom = int((kernel.shape[1] -1) -cy)
    bordered_img = cv2.copyMakeBorder(img, top=-ktop, bottom=kbottom, left=-kleft, right=kright, borderType=cv2.BORDER_CONSTANT)
    out = np.zeros((img.shape[0],img.shape[1]))
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            for x in range(ktop,kbottom+1):
                for y in range(kleft,kright+1):
                    out[i,j] +=  bordered_img[(i-ktop)+x,(j-kleft)+y] * kernel[(kernel.shape[0]-1)-(x-ktop),(kernel.shape[1]-1)-(y-kleft)]
    return out
    
    
def zero_crossing(img):
    out=np.zeros((img.shape[0],img.shape[1]))
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            if np.sign(img[i,j-1]) != np.sign(img[i,j+1]):
                out[i,j] = img[i,j]
            if np.sign(img[i-1,j]) != np.sign(img[i+1,j]):
                out[i,j] = img[i,j]
    return out

def thresholding(img,zero):
    for i in range(zero.shape[0]):
        for j in range(zero.shape[1]):
            local_region = img[max(0,i-1):min(zero.shape[0]-1,i+1),max(0,j-1):min(zero.shape[1]-1,j+1)]
            std=np.std(local_region)
            if(zero[i,j]>=std):
                zero[i,j] = 255
            else:
                zero[i,j] = 0
    return zero

img = cv2.imread("Lena.jpg",cv2.IMREAD_GRAYSCALE)

sigma = float(input("Enter the value of sigma: "))
kernel = get_log_kernel(sigma)

output_LoG = Convolution(img,kernel)
output_log2= output_LoG.copy()
cv2.imshow("input_image",img)
cv2.waitKey(0)

cv2.normalize(src=output_log2,dst=output_log2,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)
output_log2 = np.round(output_log2).astype(dtype=np.uint8)

cv2.imshow("laplacian_image",output_log2)
cv2.waitKey(0)

zero_crossing_img = zero_crossing(output_LoG)

cv2.imshow("zero_crossed_img",zero_crossing_img)
cv2.waitKey(0)

threshold_img = thresholding(img,zero_crossing_img)
cv2.imshow("Threshold_img",threshold_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

