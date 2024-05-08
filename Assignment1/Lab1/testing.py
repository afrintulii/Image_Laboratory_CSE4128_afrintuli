
import cv2
import numpy as np
import math

img=cv2.imread('D:\Image Lab\Bishmillah\Assignment1\Lab1\Lena.jpg',cv2.IMREAD_GRAYSCALE)

kernel_v=np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])
kernel_y= np.array([[-1,-2, -1], [0, 0, 0], [1, 2, 1]])

def find_difference(image1, image2):
    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
    difference = cv2.absdiff(image1, image2)
    difference = normalize(difference)
    
    return difference

def normalize(image):
    cv2.normalize(image,image,0,255,cv2.NORM_MINMAX)
    return np.round(image).astype(np.uint8)

def pad_image(image, kernel_height, kernel_width, kernel_center):
    pad_top = kernel_center[0]
    pad_bottom = kernel_height - kernel_center[0] - 1
    pad_left = kernel_center[1]
    pad_right = kernel_width - kernel_center[1] - 1
    
    padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values = 0)
    return padded_image

def convolution(image, kernel, kernel_center = (-1,-1)):    
    kernel_height, kernel_width = len(kernel), len(kernel[0])
    
    # if kernel center is not defined, then it will use the center symmetric center
    if kernel_center[0] == -1:
        kernel_center = ( kernel_height // 2, kernel_width // 2 )
    
    # pad the input image based on kernel and center
    padded_image = pad_image(image = image,  kernel_height = kernel_height, kernel_width = kernel_width, kernel_center = kernel_center)

    # generating output with dummy zeros(0)
    output = np.zeros_like(padded_image, dtype='float32')
    
    #print("Padded image")
    #print(padded_image)
    
    # xx = 1
    # yy = 2
    # print(f"Value at ({xx},{yy}) is {padded_image[xx,yy]}")
    
    # padded image height, width
    padded_height, padded_width = padded_image.shape

    kcx = kernel_center[0]
    kcy = kernel_center[1]
    
    # iterating through height. For (1,1) kernel, it iterates from 1 to (h - 1)
    for x in range( kcx, padded_height - ( kernel_height - (kcx+1)) ):
        # iterate through width. For (1,1) kernel, it iterates from 1 to (w - 1)
        for y in range( kcy, padded_width - ( kernel_width - (kcy + 1)) ):
            
            # calculating the portion in image, that will be convoluted now
            image_start_x = x - kcx
            image_start_y = y - kcy
            
            # if x == 1 and y == 2:
            #     print(f"For position({x},{y}): image from: ({image_start_x},{image_start_y}) to ({image_start_x+kernel_height},{image_start_y+kernel_width})")
            
            sum = 0
            NX = kernel_height // 2
            NY = kernel_width // 2
            for kx in range( -NX, NX+1):
                for ky in range( -NY, NY+1 ):
                    rel_pos_in_kernel_x = kx + NX # x-i
                    rel_pos_in_kernel_y = ky + NY # y-j
                    
                    rel_pos_in_image_x = NX - kx # 2
                    rel_pos_in_image_y = NY - ky # 2
                    
                    act_pos_in_image_x = rel_pos_in_image_x + image_start_x # 2 + 2 = 4
                    act_pos_in_image_y = rel_pos_in_image_y + image_start_y # 3 + 2 = 5
                    
                    # if( rel_pos_in_kernel_x >= kernel_height or rel_pos_in_kernel_y >= kernel_width):
                    #     print("Outside")
                    #     print(rel_pos_in_kernel_x, rel_pos_in_kernel_y)
                    #     print(kernel)
                    
                    k_val = kernel[ rel_pos_in_kernel_x ][ rel_pos_in_kernel_y ]
                    i_val = padded_image[ act_pos_in_image_x ][ act_pos_in_image_y ]
                    
                    # if x == 1 and y == 2:
                    #     #print(k_val, "*", i_val)
                    #     print(f"({rel_pos_in_image_x}, {rel_pos_in_image_y}) * ({rel_pos_in_kernel_x}, {rel_pos_in_kernel_y}): {k_val} * {i_val} Actual pos in image: ({act_pos_in_image_x}, {act_pos_in_image_y})")
                    
                    sum +=  k_val * i_val
            output[x,y] = sum

    # print("Output before cropping")
    # print(output)
    # Crop the output to the original image size
    out = output[kernel_center[0]:-kernel_height + kernel_center[0] + 1, kernel_center[1]:-kernel_width + kernel_center[1] + 1]
    
    return out


out1=convolution(img, kernel_y)
out2=convolution(img,kernel_v)
out= out1+out2

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