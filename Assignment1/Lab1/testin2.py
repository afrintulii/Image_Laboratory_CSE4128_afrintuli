# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 15:35:13 2024

@author: Asus
"""

import cv2
import numpy as np
import math

img=cv2.imread('D:\Image Lab\Bishmillah\Assignment1\Lab1\Lena.jpg',cv2.IMREAD_GRAYSCALE)

kernel=np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])
kernelj= np.array([[-1,-2, -1], [0, 0, 0], [1, 2, 1]])
p=1;
q=1;
qqq=1
if qqq==1:
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






# image = np.array([
#     [1,2,3],
#     [4,5,6],
#     [7,8,9]
# ])

# kernel = np.array([
#     [1,2,3],
#     [4,5,6],
#     [7,8,9]
# ])

# kernel = np.array([
#     [0,0,0],
#     [0,1,0],
#     [0,0,0]
# ])

#out = convolve(image=image, kernel=kernel,kernel_center=(-1,-1))
#print(out)
    
cv2.imshow("input", img)
cv2.imshow("result",out)
cv2.waitKey(0)
cv2.destroyAllWindows()