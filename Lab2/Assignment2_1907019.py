# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 01:58:49 2024

@author: Asus
"""

import numpy as np
import cv2

def split_image(image, threshold):
    b1,g1,r1 = cv2.split(image)
    std_b1 = np.std(b1)
    std_g1=np.std(g1)
    std_r1=np.std(r1)
    if (std_b1 < threshold and std_g1 < threshold and std_r1 < threshold ) or image.shape[0] <= 2 or image.shape[1] <= 2:
        mean_b1 = np.mean(b1)
        mean_g1 = np.mean(g1)
        mean_r1=np.mean(r1)
        mean_blue=np.full_like(b1,mean_b1)
        mean_green=np.full_like(g1,mean_g1)
        mean_red=np.full_like(r1,mean_r1)
        return cv2.merge((mean_blue,mean_green,mean_red))
    else:
        height, width,channels = image.shape
        mid_height = height // 2
        mid_width = width // 2
        top_left_image = split_image(image[:mid_height, :mid_width], threshold)
        top_right_image = split_image(image[:mid_height, mid_width:], threshold)
        bottom_left_image = split_image(image[mid_height:, :mid_width], threshold)
        bottom_right_image = split_image(image[mid_height:, mid_width:], threshold)
        top_half_image = np.hstack((top_left_image, top_right_image))
        bottom_half_image = np.hstack((bottom_left_image, bottom_right_image))
        return np.vstack((top_half_image, bottom_half_image))

def main():
    image = cv2.imread('Lena.jpg', cv2.IMREAD_COLOR)
    threshold = int(input("Enter the threshold for homogeneity: "))
    merged_image = split_image(image, threshold)
    cv2.imshow('Merged Image', merged_image)
    cv2.imshow("Input Image",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

