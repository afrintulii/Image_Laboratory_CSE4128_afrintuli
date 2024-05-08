# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 01:36:47 2024

@author: Asus
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def get_log_kernel(sigma):
    n=int(sigma*7) 
  
    kernel=np.zeros((n,n))
    constant = -1/(np.pi* sigma **4)
    center =  n//2
    for x in range (n):
        for  y in range (n):
            dx = x -center
            dy = y - center
            value= (dx**2 +dy**2)/(2*(sigma**2))
            
            kernel[x,y]=constant*(1-value)* np.exp(-value)
    return kernel


def convolution(img,kernel):
    
    height,width = kernel.shape
    padding_x,padding_y= height//2 ,width//2
    bordered_img = cv2.copyMakeBorder(img, top=padding_x, bottom=padding_x, left=padding_y, right=padding_y, borderType=cv2.BORDER_REPLICATE)
    
    result_img =np.zeros((img.shape[0],img.shape[1]),dtype="float32")
    
    #convolution Operation
    for i in range (padding_x,bordered_img.shape[0]-padding_x):
        for j in range(padding_y,bordered_img.shape[1]-padding_y):
            sum=0
            for p in range(-padding_x,padding_x+1):
                for q in range(-padding_y,padding_y+1):
                    sum+=kernel[p+padding_x][q+padding_y]*bordered_img[i-p][j-q]
            result_img[i-padding_x][j-padding_y] = sum
            
    cv2.normalize(result_img,result_img,0,255,cv2.NORM_MINMAX)
    result_img = np.round(result_img).astype(np.uint8)
    return result_img
                    

def applyZeroCrossing(img):
    
    out = np.zeros((img.shape[0], img.shape[1]))
    
    for i in range(0 + 1, img.shape[0] - 1):
        for j in range(0 + 1, img.shape[1] - 1):
            if np.sign(img[i, j-1]) != np.sign(img[i, j+1]):
                out[i, j] = img[i, j]
            if np.sign(img[i-1, j]) != np.sign(img[i+1, j]):
                out[i, j] = img[i, j]
    
    return out
           

def main():
    img = cv2.imread("Lena.jpg",cv2.IMREAD_GRAYSCALE)
    sigma=float(input("Enter The value of Sigma: "))
    kernel=get_log_kernel(sigma)
    convoluted_img = convolution(img,kernel)
    convoluted_img2 = convoluted_img.copy()
    cv2.normalize(convoluted_img,convoluted_img,0,255,cv2.NORM_MINMAX)
    convoluted_img = np.round(convoluted_img).astype(np.uint8)
    cv2.imshow("convoluted image",convoluted_img2)
    cv2.waitKey(0)
    zero_crossing_img= applyZeroCrossing(convoluted_img2)
    cv2.imshow("zero_crossing_img",zero_crossing_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()
            