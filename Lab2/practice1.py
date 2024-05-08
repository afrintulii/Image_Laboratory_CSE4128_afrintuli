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
        # Replace the image with its mean
        #b1,g1,r1 = cv2.split(image)
        mean_b1 = np.mean(b1)
        mean_g1 = np.mean(g1)
        mean_r1=np.mean(r1)
        mean_blue=np.full_like(b1,mean_b1)
        mean_green=np.full_like(g1,mean_g1)
        mean_red=np.full_like(r1,mean_r1)
        return cv2.merge((mean_blue,mean_green,mean_red))
    else:
        height, width = image.shape[:2]
        mid_h, mid_w = height // 2, width // 2
        top_left_img = split_image(image[:mid_h, :mid_w], threshold)
        top_right_img = split_image(image[:mid_h, mid_w:], threshold)
        bottom_left_img = split_image(image[mid_h:, :mid_w], threshold)
        bottom_right_img = split_image(image[mid_h:, mid_w:], threshold)
        # Merge the split images
        #cv2.imshow("top_left_img",top_left_img)
        #cv2.imshow()
        top_half = np.hstack((top_left_img, top_right_img))
        bottom_half = np.hstack((bottom_left_img, bottom_right_img))
        return np.vstack((top_half, bottom_half))

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

