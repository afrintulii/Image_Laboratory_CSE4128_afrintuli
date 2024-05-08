# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 19:43:07 2024

@author: Asus
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def equalized_histogram(s,image):
    histogram=np.zeros(256,dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel=image[i,j]
            histogram[pixel]+=1
    pdf=np.zeros(256,dtype=np.float32)
    m,n=image.shape
    size=m*n
    for i in range(0,256):
        pdf[i]=histogram[i]/size
    cdf=np.zeros(256,dtype=np.float32)
    cdf[0]= pdf[0]
    for i in range(1,256):
        cdf[i]=cdf[i-1]+pdf[i]
    for i in range(0,256):
        cdf[i] = round(cdf[i]*255)
    
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,10))
    ax1.set_title(s+'pdf')
    ax1.plot(pdf)
    ax2.set_title(s+'cdf')
    ax2.plot(cdf)
    plt.show()
    equalized_image = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            equalized_image[i,j] = cdf[image[i,j]]
    return equalized_image
    

col_img = cv2.imread('color_img.jpg',cv2.IMREAD_COLOR)
b,g,r=cv2.split(col_img)
color = ('b','g','r')
plt.figure(1,figsize=(20,10))
for i,col in enumerate(color):
    plt.subplot(2,3,i+1)
    histr,_ = np.histogram(col_img[:,:,i],256,[0,256])
    plt.plot(histr,color=col)
    plt.title('channel'+str(1+i))
plt.show()    


b_equalized = equalized_histogram("blue_rgb_image", b)   
g_equalized = equalized_histogram("green_rgb_image", g)
r_equalized = equalized_histogram("red_rgb_image", r)

merge_image = cv2.merge((b_equalized,g_equalized,r_equalized))
cv2.imshow("input_rgb_image",col_img)
cv2.imshow("b_equalized",b_equalized)
cv2.imshow("g_equalized",g_equalized)
cv2.imshow("r_equalized",r_equalized)
cv2.imshow("equalized_col_image",merge_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.figure(2,figsize=(20,10))
for i,col in enumerate(color):
    plt.subplot(2,3,i+1)
    histr,_ = np.histogram(merge_image[:,:,i],256,[0,256])
    plt.plot(histr,color=col)
    plt.title('channel'+str(1+i))
plt.show()

equalized_b= equalized_histogram("blue_equalized_image", b_equalized)
equalized_g= equalized_histogram("green_equalized_image", g_equalized)
equalized_r= equalized_histogram("red_equalized_image", r_equalized)

hsv_img = cv2.cvtColor(col_img, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv_img)
color = ('b','g','r')
plt.figure(3,figsize=(20,10))
for i,col in enumerate(color):
    plt.subplot(2,3,i+1)
    histr,_ = np.histogram(hsv_img[:,:,i],256,[0,256])
    plt.plot(histr,color=col)
    plt.title('channel'+str(1+i))
plt.show()
v_equalized = equalized_histogram("value_hsv_image", v)
merge_image = cv2.merge((h,s,v_equalized))
merge_image = cv2.cvtColor(merge_image, cv2.COLOR_HSV2BGR)
cv2.imshow("input_hsv_image",hsv_img)
cv2.imshow("h_hsv",h)
cv2.imshow("s_hsv",s)
cv2.imshow("v_equalized",v_equalized)
cv2.imshow("equalized_hsv_image",merge_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.figure(4,figsize=(20,10))
for i,col in enumerate(color):
    plt.subplot(2,3,i+1)
    histr,_ = np.histogram(merge_image[:,:,i],256,[0,256])
    plt.plot(histr,color=col)
    plt.title('channel'+str(1+i))
plt.show()

equalized_v= equalized_histogram("value_equalized_image", v_equalized)


