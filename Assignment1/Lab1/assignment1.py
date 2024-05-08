# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 01:48:35 2024

@author: Asus
"""

import cv2
import numpy as np
import math

def main():
    print("Assignment1_1907019")
    print("1.GrayScale Image:")
    print("2.RGB Image:")
    print("3.HSV Image:")
    print("4:Show Difference Between Gaussian Concolution of RGB and HSV")
    choice = input("Enter Your Choice: ")
    choice = int(choice)
    if choice == 1:
        image=cv2.imread('Lena.jpg',cv2.IMREAD_GRAYSCALE)
        image2=cv2.imread('noisy_image.jpg',cv2.IMREAD_GRAYSCALE)
        #cv2.imshow('GrayScale',image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    elif choice==2:
        image=cv2.imread('Lena.jpg',cv2.IMREAD_COLOR)
        image2=cv2.imread('noisy_image.jpg',cv2.IMREAD_COLOR)
        #cv2.imshow('RGB',image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    elif choice==3:
        image=cv2.imread('Lena.jpg',cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image2=cv2.imread('noisy_image.jpg',cv2.COLOR_RGB2HSV)
        #cv2.imshow('HSV',image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    elif choice ==4:
       image=cv2.imread('noisy_image.jpg',cv2.IMREAD_COLOR)
       image2 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
       
       
    
    image=cv2.resize(image,(500,500))
    imag2=cv2.resize(image,(500,500))
    print("******Smoothing Filter******")
    print("1.Gaussian Blur Filter")
    print("2.Mean Filter")
    print("*****Sharpening Filter******")
    print("3.Laplacian Filter")
    print("4.Laplacian of a Gaussian(LOG)")
    print("5.Sobel Filter")
    #print("Enter Your Choice: ")
    filter_str=input("Enter Your Choice: ")
    filter=int(filter_str)
    if filter == 1:
        sigmax=float(input("Enter the value of sigmax: "))
        sigmay=float(input("Enter the value of sigmay: "))
        kernel=get_gaussian_kernel(sigmax, sigmay)
        center_x=int(input("Kernel center_x: "))
        center_y=int(input("Kernel center_y: "))
        if choice == 1:
            result=convolution("grayscale",image,kernel,center_x,center_y)
            cv2.imshow("GrayScale_Image",image)
            cv2.imshow("Gaussian Filtered Image",result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif choice == 2:
            b1,g1,r1=cv2.split(image)
            b1=convolution("blue",b1,kernel,center_x,center_y)
            g1=convolution("green",g1,kernel,center_x,center_y)
            r1=convolution("red",r1,kernel,center_x,center_y)
            merged=cv2.merge((b1,g1,r1))
            cv2.imshow("RGB Image",image)
            cv2.imshow("blue",b1)
            cv2.imshow("green",g1)
            cv2.imshow("red",r1)  
            cv2.imshow("merged",merged)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif choice ==3 :
            b1,g1,r1=cv2.split(image)
            b1=convolution("blue_hsv",b1,kernel,center_x,center_y)
            g1=convolution("green_hsv",g1,kernel,center_x,center_y)
            r1=convolution("red_hsv",r1,kernel,center_x,center_y)
            merged=cv2.merge((b1,g1,r1))
            cv2.imshow("HSV Image",image)
            cv2.imshow("blue",b1)
            cv2.imshow("green",g1)
            cv2.imshow("red",r1)
            cv2.imshow("merged",merged)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif choice == 4 :
            b1,g1,r1=cv2.split(image)
            b1=convolution("blue",b1,kernel,center_x,center_y)
            g1=convolution("green",g1,kernel,center_x,center_y)
            r1=convolution("red",r1,kernel,center_x,center_y)
            merged=cv2.merge((b1,g1,r1))
            merged=cv2.resize(merged,(512,512))
            b2,g2,r2=cv2.split(image2)
            b1=convolution("blue_hsv",b1,kernel,center_x,center_y)
            g1=convolution("green_hsv",g1,kernel,center_x,center_y)
            r1=convolution("red_hsv",r1,kernel,center_x,center_y)
            merged2=cv2.merge((b2,g2,r2))
            merged2=cv2.resize(merged2,(512,512))
            convolved_image_hsv_rgb = cv2.cvtColor(merged2, cv2.COLOR_HSV2RGB)
            convolved_image_rgb_float = merged.astype(np.float32)
            convolved_image_hsv_rgb_float = merged2.astype(np.float32)
            subtracted_image = convolved_image_hsv_rgb_float - convolved_image_rgb_float
            cv2.normalize(subtracted_image, subtracted_image, 0, 255, cv2.NORM_MINMAX)
            subtracted_image = np.round(subtracted_image).astype(np.uint8)
            cv2.imshow("RGB Image",image)
            cv2.imshow("HSV Image",image2)
            cv2.imshow("RGB Convoluted Image",merged)
            cv2.imshow("HSV Convoluted Image",merged2)
            cv2.imshow("Subtracted_Image",subtracted_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()            

            
        
    elif filter == 2:
        print("Enter Your Kernel Hight and Width: ")
        x=input("height: ")
        x=int(x)
        y=input("Width:  ")
        y=int(y)
        kernel=get_mean_kernel(x,y)
        center_x=int(input("Kernel center_x: "))
        center_y=int(input("Kernel center_y: "))
        if choice == 1:
            result=convolution("grayscale",image,kernel,center_x,center_y)
            cv2.imshow("GrayScale_Image",image)
            cv2.imshow("Mean Filtered Image",result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif choice == 2:
            b1,g1,r1=cv2.split(image)
            b1=convolution("blue",b1,kernel,center_x,center_y)
            g1=convolution("green",g1,kernel,center_x,center_y)
            r1=convolution("red",r1,kernel,center_x,center_y)
            merged=cv2.merge((b1,g1,r1))
            cv2.imshow("RGB Image",image)
            cv2.imshow("blue",b1)
            cv2.imshow("green",g1)
            cv2.imshow("red",r1)
            
            cv2.imshow("merged",merged)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif choice ==3 :
            b1,g1,r1=cv2.split(image)
            b1=convolution("blue_hsv",b1,kernel,center_x,center_y)
            g1=convolution("green_hsv",g1,kernel,center_x,center_y)
            r1=convolution("red_hsv",r1,kernel,center_x,center_y)
            merged=cv2.merge((b1,g1,r1))
            cv2.imshow("HSV Image",image)
            cv2.imshow("blue",b1)
            cv2.imshow("green",g1)
            cv2.imshow("red",r1)
            cv2.imshow("merged",merged)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif filter ==3:
        print("Enter Your Kernel Hight and Width: ")
        x=input("height: ")
        x=int(x)
        kernel=get_laplacian_kernel(x)
        
        center_x=int(input("Kernel center_x: "))
        center_y=int(input("Kernel center_y: "))
        if choice == 1:
            result=convolution("grayscale",image,kernel,center_x,center_y)
            cv2.imshow("GrayScale_Image",image)
            cv2.imshow("laplacian Filtered Image",result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif choice ==2 :
            b1,g1,r1=cv2.split(image)
            b1=convolution("blue",b1,kernel,center_x,center_y)
            g1=convolution("green",g1,kernel,center_x,center_y)
            r1=convolution("red",r1,kernel,center_x,center_y)
            merged=cv2.merge((b1,g1,r1))
            cv2.imshow("RGB Image",image)
            cv2.imshow("blue",b1)
            cv2.imshow("green",g1)
            cv2.imshow("red",r1)
            cv2.imshow("merged",merged)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif choice ==3 :
            b1,g1,r1=cv2.split(image)
            b1=convolution("blue_hsv",b1,kernel,center_x,center_y)
            g1=convolution("green_hsv",g1,kernel,center_x,center_y)
            r1=convolution("red_hsv",r1,kernel,center_x,center_y)
            merged=cv2.merge((b1,g1,r1))
            cv2.imshow("HSV Image",image)
            cv2.imshow("blue",b1)
            cv2.imshow("green",g1)
            cv2.imshow("red",r1)
            cv2.imshow("merged",merged)
            cv2.waitKey(0)
            cv2.destroyAllWindows()            
    elif filter == 4:
        x=float(input("Enter The value of Sigma: "))
        kernel=get_log_kernel(x)
            
        
        center_x=int(input("Kernel center_x: "))
        center_y=int(input("Kernel center_y: "))
        if choice == 1:
            result=convolution("grayscale",image,kernel,center_x,center_y)
            cv2.imshow("GrayScale_Image",image)
            cv2.imshow("laplacian Filtered Image",result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif choice == 2:
            b1,g1,r1=cv2.split(image)
            b1=convolution("blue",b1,kernel,center_x,center_y)
            g1=convolution("green",g1,kernel,center_x,center_y)
            r1=convolution("red",r1,kernel,center_x,center_y)
            merged=cv2.merge((b1,g1,r1))
            cv2.imshow("blue",b1)
            cv2.imshow("green",g1)
            cv2.imshow("red",r1)
            cv2.imshow("merged",merged)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif choice == 3:
            b1,g1,r1=cv2.split(image)
            b1=convolution("blue_hsv",b1,kernel,center_x,center_y)
            g1=convolution("green_hsv",g1,kernel,center_x,center_y)
            r1=convolution("red_hsv",r1,kernel,center_x,center_y)
            merged=cv2.merge((b1,g1,r1))
            cv2.imshow("blue",b1)
            cv2.imshow("green",g1)
            cv2.imshow("red",r1)
            cv2.imshow("merged",merged)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
          
                   
    elif filter == 5:
        
        h_v=int(input("Enter Your Choice: "))
        kernel=get_sobel_kernel(h_v)
        center_x=int(input("kernel center_x: "))
        center_y=int(input("kernel center_y: "))
        if choice == 1:
            result=convolution("grayscale",image,kernel,center_x=1,center_y=1)
            cv2.imshow("GrayScale_Image",image)
            cv2.imshow("Sobel_Result",result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif choice == 2:
            b1,g1,r1=cv2.split(image)
            b1=convolution("blue",b1,kernel,center_x,center_y)
            g1=convolution("green",g1,kernel,center_x,center_y)
            r1=convolution("red",r1,kernel,center_x,center_y)

            merged=cv2.merge((b1,g1,r1))
            cv2.imshow("blue",b1)
            cv2.imshow("green",g1)
            cv2.imshow("red",r1)
            cv2.imshow("merged",merged)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif choice == 3:
            b1,g1,r1=cv2.split(image)
            b1=convolution("blue_hsv",b1,kernel,center_x,center_y)
            g1=convolution("green_hsv",g1,kernel,center_x,center_y)
            r1=convolution("red_hsv",r1,kernel,center_x,center_y)
            merged=cv2.merge((b1,g1,r1))
            cv2.imshow("merged",merged)
            cv2.waitKey(0)
            cv2.destroyAllWindows()               
        
#Defining Kernel    
       
def get_mean_kernel(height,width):
    mean_kernel=np.ones((height,width),dtype=np.float32)
    mean_kernel=mean_kernel/(height*width)
    return mean_kernel

def get_gaussian_kernel(sigmax,sigmay):
    height=int(sigmax*5)
    width=int(sigmay*5)
    if height%2==0:
        height+=1
    if width%2==0:
        width+=1
    gaussian_kernel=np.zeros((height,width),dtype=np.float32)
    constant=1/(2*math.pi*(sigmax*sigmay))
    height=height//2
    width=width//2
    
    for x in range(-height,height+1):
        for y in range(-width,width+1):
            exponent=-0.5*((x*x)/(sigmax*sigmax)+(y*y)/(sigmay*sigmay))
            value=constant*math.exp(exponent)
    
            gaussian_kernel[x+height][y+width]=value
    
    return gaussian_kernel

def get_laplacian_kernel(x):
        kernel_3=np.array([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]], dtype=np.float32)
        kernel_5=np.array([ [0, 0,  1,  0, 0],
                            [0, 1,  2,  1, 0],
                            [1, 2, -16, 2, 1],
                            [0, 1,  2,  1, 0],
                            [0, 0,  1,  0, 0]], dtype=np.float32)
        if x==3:
            return kernel_3
        elif x==5:
            return kernel_5
 
def get_sobel_kernel(h_v):
    if h_v == 1:
        kernel=np.array([[-1, 0, 1],
                             [-2, 0, 2], 
                             [-1, 0, 1]])
        
    elif h_v == 2:
        kernel=np.array([[-1,-2, -1], 
                                 [0, 0, 0], 
                                 [1, 2, 1]])

    return kernel    
    


def get_log_kernel(sigma):
    n = int(sigma * 7)
    n = n | 1
    
    kernel = np.zeros( (n,n) )

    center = n // 2
    part1 = -1 / (np.pi * sigma**4)
    
    for x in range(n):
        for y in range(n):
            dx = x - center
            dy = y - center
            
            part2 = (dx**2 + dy**2) / (2 * sigma**2)
            
            kernel[x][y] =  part1 * (1 - part2) * np.exp(-part2)

    
    return (kernel)

#Convolution Code
def convolution(s,image,kernel,center_x,center_y):
    k = kernel.shape[0] // 2
    l = kernel.shape[1] // 2
    padding_bottom = kernel.shape[0] - 1 - center_x
    padding_right = kernel.shape[1] - 1 - center_y
    img_bordered = cv2.copyMakeBorder(src=image, top=center_x, bottom=padding_bottom, left=center_y, right=padding_right,borderType=cv2.BORDER_CONSTANT)
    out = np.zeros((img_bordered.shape[0],img_bordered.shape[1]),dtype=np.uint8)

    for i in range(center_x, img_bordered.shape[0] - padding_bottom - k):
        for j in range(center_y, img_bordered.shape[1] - padding_right - l):
            res = 0
            for x in range(-k, k + 1):
                for y in range(-l, l + 1):
                    res += kernel[x + k, y + l] * img_bordered[i - x, j - y]
            out[i, j] = res 
            
    # crop image to original image
    #out = out[center_x: -padding_bottom, center_y:-padding_right]
    cv2.normalize(out, out, 0, 255, cv2.NORM_MINMAX)
    out = np.round(out).astype(np.uint8)
    #print(f"normalized {out}") 
    #out = out[p: -padding_bottom, q:-padding_right]
           
    return out
    
 
    
    
if __name__ == "__main__":
    main()
    
    
    