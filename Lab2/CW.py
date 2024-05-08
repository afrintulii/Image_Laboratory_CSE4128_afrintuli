# -*- coding: utf-8 -*-
# -- coding: utf-8 --
"""
Created on Wed Feb 28 11:33:03 2024

@author: Hp
"""

import numpy as np
import cv2


def get_log_kernel(sigma, MUL = 7):
    
    n = int(sigma * MUL)
    n = n | 1
    kernel = np.zeros( (n,n) )
    center = n // 2
    part1 = -1 / (np.pi * sigma**4)
    for x in range(n):
       for y in range(n):
           dx = x - center
           dy = y - center
           part2 = (dx*2 + dy*2) / (2 * sigma*2)
           kernel[x][y] =  part1 * (1 - part2) * np.exp(-part2)
    return kernel

def convolution(image,kernel):
    v=(kernel.shape[0]-1)//2
    mimg=cv2.copyMakeBorder(img,v , v, v, v, cv2.BORDER_CONSTANT)
    result=np.zeros((image.shape[0],image.shape[1]),dtype="float32")
    for i in range(v,mimg.shape[0]-v):
        for j in range(v,mimg.shape[1]-v):
            sum=0
            for p in range(-v,v+1):
                for q in range(-v,v+1):
                    sum=sum+kernel[p+v][q+v]*mimg[i-p][j-q]
            
            result[i-v][j-v]=sum
    return result



            
    


def zero_crossing(image):
    zero_crossings = np.zeros_like(image, dtype=np.uint8)
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            #if image[i][j] == 0:
                #continue
            if (image[i+1][j] > 0 and image[i-1][j] < 0) or (image[i+1][j] < 0 and image[i-1][j] > 0):
                zero_crossings[i][j] = 255
            elif (image[i][j+1] > 0 and image[i][j-1] < 0) or (image[i][j+1] < 0 and image[i][j-1] > 0):
                zero_crossings[i][j] = 255
                
            
    return zero_crossings
def variance_thresholding(image,i,j,threshold,kernel):
    padding=kernel.shape[0]//2
    # Calculate the variance of the image
    loc_reg=image[i-padding:i+padding+1,j-padding:j+padding+1]
    local_std=np.std(loc_reg)
    # Thresholding
    if local_std > threshold:
        return 255
    else:
        return 0



    
    
                    

img=cv2.imread("Lena.jpg",cv2.IMREAD_GRAYSCALE)
img=cv2.resize(img,(500,500))

sigma=1

kernel=get_log_kernel(sigma)
out=np.zeros((img.shape[0],img.shape[1]),dtype="float32")
out=convolution(img,kernel)
out1=np.zeros((out.shape[0],out.shape[1]),dtype="float32")
cv2.normalize(out,out1,0,255,cv2.NORM_MINMAX)
out1=np.round(out1).astype(np.uint8)
cv2.imshow("Gray",img)
cv2.imshow("convolved", out1)
zero_out=zero_crossing(out)
thresholded_image = np.zeros_like(zero_out)
for i in range(zero_out.shape[0]):
    for j in range(zero_out.shape[1]):
        thresholded_image[i][j] = variance_thresholding(zero_out,i,j,100,kernel)
#th_out=thresholding(img,zero_out,kernel)
cv2.imshow("Zero crossing",zero_out)
cv2.imshow("Output",thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

