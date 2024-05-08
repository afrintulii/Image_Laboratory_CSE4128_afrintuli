# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 01:50:48 2024

@author: Asus
"""

import numpy as np
import cv2
import math


def main():
    print("Assignment 1")
    color_mode = input("Enter the colormode \n RGB(1) \n HSV(0) :")
    color_mode = int(color_mode)
    if color_mode == 1:
        img = cv2.imread("D:\Image Lab\Bishmillah\Assignment1\Lab1\Lena.jpg",cv2.IMREAD_COLOR)
    elif color_mode == 0:
        img = cv2.imread("D:\Image Lab\Bishmillah\Assignment1\Lab1\Lena.jpg",cv2.COLOR_RGB2HSV)
    
    
    img = cv2.resize(img, (500,500))
    
    
    apply_filter_str = input("Enter the filter code \n Smoothing Filter: \n 1. Gaussian Filter\n 2. Mean Filter \n Sharpening Filter :\n 3. Laplacian Filter \n 4. LoG  \n 5. Sobel filter  : ")
    apply_filter = int(apply_filter_str)
    if apply_filter == 1:
        
        x=input("height:")
        x=int(x)
        y=input("width:")
        y=int(y)
        kernel = get_gaussian_kernel(x,y)
        center_x=int(input("kernel center_x:"))
        center_y=int(input("kernel_center_y:"))
        b1, g1, r1 = cv2.split(img)
        b1 = convolution("blue",b1, kernel,center_x,center_y)
        g1 = convolution("green",g1 , kernel,center_x,center_y)
        r1 = convolution("red",r1 , kernel,center_x,center_y)
        merged = cv2.merge((b1, g1, r1))
        print(b1)
        cv2.imshow("merged", merged)
        

    elif apply_filter == 2:
        x=input("height:")
        x=int(x)
        y=input("width:")
        y=int(y)
        kernel = get_mean_kernel(x,y)
        center_x=int(input("kernel center_x:"))
        center_y=int(input("kernel_center_y:"))
        b1, g1, r1 = cv2.split(img)
        b1 = convolution("blue",b1, kernel,center_x,center_y)
        g1 = convolution("green",g1 , kernel,center_x,center_y)
        r1 = convolution("red",r1 , kernel,center_x,center_y)
        merged = cv2.merge((b1, g1, r1))
        print(b1)
        cv2.imshow("merged", merged)
        
        
    elif apply_filter == 3:
        x=input("size:")
        x=int(x)
        y=input("1.true \n 0.false:")
        y=int(y)
        kernel = get_laplacian_kernel(x,y)
        center_x=int(input("kernel center_x:"))
        center_y=int(input("kernel_center_y:"))
        b1, g1, r1 = cv2.split(img)
        b1 = convolution("blue",b1, kernel,center_x,center_y)
        g1 = convolution("green",g1 , kernel,center_x,center_y)
        r1 = convolution("red",r1 , kernel,center_x,center_y)
        merged = cv2.merge((b1, g1, r1))
        print(b1)
        cv2.imshow("merged", merged)
        
    elif apply_filter == 4:
        x=input("size:")
        x=int(x)
        y=input("sigma:")
        y=int(y)
        kernel = get_log_kernel(x,y)
        center_x=int(input("kernel center_x:"))
        center_y=int(input("kernel_center_y:"))
        b1, g1, r1 = cv2.split(img)
        b1 = convolution("blue",b1, kernel,center_x,center_y)
        g1 = convolution("green",g1 , kernel,center_x,center_y)
        r1 = convolution("red",r1 , kernel,center_x,center_y)
        merged = cv2.merge((b1, g1, r1))
        print(b1)
        cv2.imshow("merged", merged)
        
    elif apply_filter == 5:
        h_v= int(input("1.Horizontal\n 2.Vertical"))
        kernel = get_sobel_kernel(h_v)
        center_x=int(input("kernel center_x:"))
        center_y=int(input("kernel_center_y:"))
        b1, g1, r1 = cv2.split(img)
        b1 = convolution("blue",b1, kernel,center_x,center_y)
        g1 = convolution("green",g1 , kernel,center_x,center_y)
        r1 = convolution("red",r1 , kernel,center_x,center_y)
        merged = cv2.merge((b1, g1, r1))
        print(b1)
        cv2.imshow("merged", merged)
        
        
    else:
        print("Invalid filter code.")
    print(kernel)
    cv2.imshow("lenna",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


#Convolution Code 

def convolution(s,img,kernel,p,q):
    k = kernel.shape[0] // 2
    l = kernel.shape[1] // 2
    padding_bottom = kernel.shape[0] - 1 - p
    padding_right = kernel.shape[1] - 1 - q
    img_bordered = cv2.copyMakeBorder(src=img, top=p, bottom=padding_bottom, left=q, right=padding_right,borderType=cv2.BORDER_CONSTANT)
    out = img_bordered.copy()

    for i in range(p, img_bordered.shape[0] - padding_bottom - k):
        for j in range(q, img_bordered.shape[1] - padding_right - l):
            res = 0
            for x in range(-k, k + 1):
                for y in range(-l, l + 1):
                    res += kernel[x + k, y + l] * img_bordered[i - x, j - y]
            out[i, j] = res    

    cv2.normalize(out, out, 0, 255, cv2.NORM_MINMAX)
    out = np.round(out).astype(np.uint8)
    print(f"normalized {out}") 
    #out = out[p: -padding_bottom, q:-padding_right]
    cv2.imshow(s, out)       
    return out
    
    
    
    
    
#definition of smoothing filters 

def get_gaussian_kernel(height, width):
    gaussian_kernel = np.zeros((height, width))
    sigma_x = float(input("Value of sigma_X: "))  # Shape and smoothness control
    sigma_y = float(input("Value of sigma_Y: "))
    constant = 1 / (2 * sigma_x * sigma_y)
    height = height // 2
    width = width // 2

    for x in range(-height, height + 1):
        for y in range(-width, width + 1):
            exponent = -0.5 * ((x * x / (sigma_x * sigma_x)) + (y * y / (sigma_y * sigma_y)))
            gaussian_value = constant * math.exp(exponent)

            gaussian_kernel[x + height, y + width] = gaussian_value

    return gaussian_kernel

def get_mean_kernel(height,width):
    mean_kernel= np.ones((height,width),dtype=np.float32)
    mean_kernel = mean_kernel / (height * width)
    return mean_kernel

def get_laplacian_kernel(size, center_coefficient):
    if center_coefficient == 1:
        coefficient=-1
    else:
        coefficient=1
    laplacian_kernel=np.ones((size,size),dtype=int)*coefficient
    center = (size*size)-1
    laplacian_kernel[size//2][size//2]= center*(-coefficient)
    print(laplacian_kernel)
    #now updating the center
    
    
    return laplacian_kernel

def get_log_kernel(size,sigma):
    kernel = np.zeros((size, size))
    c = (1 / (2 * np.pi * sigma ** 2))
    size = size // 2

    for x in range(-size, size + 1):
        for y in range(-size, size + 1):
            kernel[x + size, y + size] = c * np.exp(-((x * x) + (y * y)) / (2 * sigma ** 2))

    gauss = kernel / np.sum(kernel)

    for x in range(-size, size + 1):
        for y in range(-size, size + 1):
            kernel[x + size, y + size] = -1 * ((x * x) + (y * y) - (2 * sigma ** 2)) * gauss[x + size, y + size]

    return kernel
    

def get_sobel_kernel(h_v):
    kernel_v = np.array([[-1, 0, 1],
                         [-2, 0, 2], 
                         [-1, 0, 1]])
    kernel_h = np.array([[-1,-2, -1], 
                         [0, 0, 0], 
                         [1, 2, 1]])
    return kernel_h if h_v else kernel_v    
    
    
if __name__ == "__main__":
    main()