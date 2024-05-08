import numpy as np 
import cv2 

def log_kernel(sigma =1 ):
    
    n = int(sigma * 7)
    kernel = np.zeros( (n,n) )
    center = n // 2
    part1 = -1 / (np.pi * sigma**4)
    for x in range(n):
       for y in range(n):
           dx = x - center
           dy = y - center
           part2 = (dx**2 + dy**2) / (2 * sigma**2)
           kernel[x][y] =  part1 * (1 - part2) * np.exp(-part2)
    return kernel


img = cv2.imread("Lenna.png",cv2.IMREAD_GRAYSCALE)

cv2.imshow("Inputimg",img)
cv2.waitKey(0)

kernel_x = log_kernel()

image_r = img.shape[0]
image_c = img.shape[1]

kernel_r = kernel_x.shape[0]
kernel_c = kernel_x.shape[1]

pad_r = kernel_r // 2
pad_c = kernel_c // 2

out = np.zeros((img.shape[0],img.shape[1]))  #output array

for x in range (pad_r ,image_r - pad_r ):
    for y in range (pad_c ,image_c - pad_c ):
        sum = 0
        for m in range ( -pad_r , pad_r+1):
            for n in range (-pad_c , pad_c +1 ):
                sum += img[x-m][y-n] * kernel_x[pad_r+m][pad_c+n]

        out[x][y] = sum


cv2.waitKey(0)      
cv2.imshow('output image',out)
print(out)

z_c_image = np.zeros(out.shape)

for i in range(0,out.shape[0]-1):
    for j in range(0,out.shape[1]-1):
            
                if (out[i+1][j] < 0 and  out[i-1][j] > 0) or (out[i+1][j] > 0 and  out[i-1][j] < 0) or (out[i][j+1] < 0 and  out[i][j-1] > 0) or (out[i][j+1] > 0 and  out[i][j-1] < 0) :
                    local_region = out[i-pad_r:i+pad_r+1, j-pad_c:j+pad_c+1]
                    local_stddev = np.std(local_region)
                    if (local_stddev**2>60):
                      z_c_image[i,j] = 1
                else:
                     z_c_image[i,j] = 0 
   
cv2.imshow('z_c image',z_c_image)


cv2.normalize(out,out, 0, 255, cv2.NORM_MINMAX)
out = np.round(out).astype(np.uint8)
print(out)
cv2.imshow('normalised output image',out)
cv2.normalize(z_c_image,z_c_image, 0, 255, cv2.NORM_MINMAX)
cv2.imshow('normalised z_c output image',z_c_image)

cv2.waitKey(0)
cv2.destroyAllWindows()