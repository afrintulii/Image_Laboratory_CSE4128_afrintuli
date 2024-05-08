# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:25:52 2024

@author: Asus
"""

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
def main():
    image=cv2.imread('Lena.jpg',cv2.IMREAD_GRAYSCALE)
    sigma=float(input("Enter Sigma: "))
    kernel=get_log_kernel(sigma)
    convolution(image,kernel,3,3)
     

def get_log_kernel(sigma):
    n = int(sigma * 7)
    n = n | 1
    
    kernel = np.zeros( (n,n) )

    center = n // 2
    part1 = -1 / (np.pi * sigma**4)
    for x in range(-center,center+1):
        for y in range(-center,center+1):
            part2 = (x**2 + y**2) / (2 * sigma**2)
    
            kernel[x+center][y+center]=part1 *(1-part2) * np.exp(-part2)
    
    return kernel
    
    

def convolution(image,kernel,center_x,center_y):
    k = kernel.shape[0] // 2
    l = kernel.shape[1] // 2
    padding_bottom = kernel.shape[0] - 1 - center_x
    padding_right = kernel.shape[1] - 1 - center_y
    img_bordered = cv2.copyMakeBorder(src=image, top=center_x, bottom=padding_bottom, left=center_y, right=padding_right,borderType=cv2.BORDER_CONSTANT)
    out = np.zeros((img_bordered.shape[0],img_bordered.shape[1]),dtype=np.uint8)
    zero_crossing=np.zeros((img_bordered.shape[0],img_bordered.shape[1]),dtype=np.uint8)
    for i in range(center_x, img_bordered.shape[0] - padding_bottom - k):
        for j in range(center_y, img_bordered.shape[1] - padding_right - l):
            res = 0
            for x in range(-k, k + 1):
                for y in range(-l, l + 1):
                    res += kernel[x + k, y + l] * img_bordered[i - x, j - y]
                    #print(res)
            out[i, j] = res 
            #print(out[i,j])
    for i in range(1,out.shape[0]-1) :
        for j in range(1,out.shape[1]-1):
            #print(out[i,j+1],out[i,j-1])
            if(out[i,j+1]>0 and out[i,j-1]<0):
                zero_crossing[i,j]=255
                #print(zero_crossing[i,j])
            elif(out[i,j+1]<0 and out[i,j-1]>0):
                zero_crossing[i,j]=255
            elif(out[i-1,j]<0 and out[i+1,j]>0):
                zero_crossing[i,j]=255
            elif(out[i-1,j]>0 and out[i+1,j]<0):
                zero_crossing[i,j]=255
            #print(zero_crossing[i,j])
            
    cv2.normalize(out, out, 0, 255, cv2.NORM_MINMAX)
    out = np.round(out).astype(np.uint8)
    out2=out.copy()
    pad=3//2 
    for i in range(1,out.shape[0]-1):
        for j in range(1,out.shape[1]-1):
            if(zero_crossing[i,j]>0):
                #print(zero_crossing[i,j])
                local_region=img_bordered[i-pad:i+pad+1,j-pad:j+pad+1]
                local_stddev=np.std(local_region)
                if(local_stddev<60):
                  zero_crossing[i,j]=0
                    
    cv2.imshow("input",image)         
    cv2.imshow("Log Image",out)
    cv2.normalize(zero_crossing,zero_crossing, 0, 255, cv2.NORM_MINMAX)
    zero_crossing= np.round(zero_crossing).astype(np.uint8)
    cv2.imshow("Thresholdingg",out2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #plt.figure()
    #plt.plot(out)
    #plt.ion()
    #plt.show()
    
if __name__ == "__main__":
    main()