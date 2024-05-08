# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:43:03 2024

@author: Asus
"""

# Fourier transform - guassian lowpass filter

import cv2
import numpy as np
from matplotlib import pyplot as plt
Do=5
n=2
def butterworth_op(filter,uk,vk,Do):
  M = filter.shape[0]
  N = filter.shape[1]
  H = np.ones((M,N), np.float32)
  for u in range(M):
      for v in range(N):
          H[u, v] = 1.0
          dk = ((u - (M/2)-uk)**2 + (v -(N/2) -vk)**2)**(0.5)
          d_k = ((u - (M/2)+uk)**2 + (v - (N/2)+vk)**2)**(0.5)
          if dk == 0 or d_k == 0:
              H[u, v] = 0.0
              continue
          H[u, v] = (1/(1+((Do/dk)**(2*n)))) * (1/(1+((Do/d_k)**(2*n))))
  return H  
    
def butterworth(filter, points,Do):
    sum = np.ones(filter.shape, np.float32)
    for u,v in points:
        sum *= butterworth_op(filter, u, v,Do)
    return sum
# take input
img_input = cv2.imread('pnois1.jpg', 0)
img = img_input.copy()
image_size = img.shape[0] * img.shape[1]
#Do=int(input("Enter The radius: "))

#%%
# fourier transform
ft = np.fft.fft2(img)
ft_shift = np.fft.fftshift(ft)
#ft_shift = ft
magnitude_spectrum_ac = np.abs(ft_shift)
magnitude_spectrum = 20 * np.log(np.abs(ft_shift)+1)
magnitude_spectrum = cv2.normalize(magnitude_spectrum, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U) 
ang = np.angle(ft_shift)
ang_ = cv2.normalize(ang, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U) 
## phase add
filter = np.ones(img.shape, float)

#points = [(262, 261),(272,256)]
points = [(262, 261)]
notch_filter = butterworth(filter, points,Do)
notch_filter_ac = 20 * np.log(np.abs(notch_filter)+1)
notch_filter_normalized = cv2.normalize(notch_filter_ac, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
result = magnitude_spectrum * notch_filter
result = 20 * np.log(np.abs(result)+1)
result_normalized = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
final_result = np.multiply(result, np.exp(1j*ang))

# inverse fourier
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
img_back_scaled = cv2.normalize(img_back, None, 0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)

## plot
cv2.imshow("input ",img)
cv2.imshow("Magnitude Spectrum", magnitude_spectrum)
cv2.imshow("Angle", ang_)
cv2.imshow("notch filter", notch_filter_normalized)
cv2.imshow("multiplied", result_normalized)
# cv2.imshow("Magnitude Spectrum normalized", magnitude_spectrum_normalized)
cv2.imshow("Inverse transform", img_back_scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()