# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:34:40 2024

@author: Asus
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 02:08:18 2024

@author: USER
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.figure(1,figsize=(20, 10))
img = cv2.imread(r"col.jpg")
cv2.imshow("input", img)
color = ('b','g','r')

for i,col in enumerate(color):   
    plt.subplot(2, 3, i+1)
    histr, _ = np.histogram(img[:,:,i],256,[0,256])
    plt.plot(histr,color = col)  #Add histogram to our plot 
    plt.title('Channel'+str(i+1))
plt.show() 

for i in range(3):
    plt.subplot(2, 3, i+4)
    plt.imshow(img[:,:,i],'gray')
    plt.title('Channel'+str(i+1))
plt.show()     


#%% HSV

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
plt.figure(2,figsize=(20, 10))
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(2, 3, 3)
histr, _ = np.histogram(img_hsv[:,:,2],256,[0,256])
plt.plot(histr,color = 'b')


plt.subplot(2, 3, 4)

img_hsv2 =img_hsv.copy()
img_hsv2[:, :, 1] = 255
img_hsv2[:, :, 2] = 255
img_hsv2 = cv2.cvtColor(img_hsv2, cv2.COLOR_HSV2RGB)
plt.imshow(img_hsv2)
plt.title('Hue Channel')

plt.subplot(2, 3, 5)
plt.imshow(img_hsv[:,:,1],'gray')
plt.title('Saturation Channel')  

plt.subplot(2,3,6)
plt.imshow(img_hsv[:,:,2],'gray')
plt.title('Value Channel')  
plt.show() 


cv2.waitKey(0)
cv2.destroyAllWindows()