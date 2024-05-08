import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def erlang_distribution(shape,scale):
    e_out=np.zeros(256,dtype=np.float32)
    mean = 1/(scale)
    c1= mean**shape
    c2= (c1) * (np.math.factorial(shape-1))
    for i in range(256):
        sum = (((i)**(shape-1)) * (np.exp(-((i)/(mean))))) / (c2)
        e_out[i] = sum
    return e_out
def histogram_equalization(img):
    histogram = np.zeros(256,dtype=np.float32)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = img[i,j]
            histogram[pixel]+=1
    pdf = np.zeros(256,dtype=np.float32)
    size = img.shape[0]*img.shape[1]
    for i in range(0,256):
        pdf[i] = histogram[i]/size
    cdf = np.zeros(256,dtype=np.float32)
    cdf[0] = pdf[0]
    new_hist =np.zeros(256,dtype=np.float32)
    for i in range(1,256):
        cdf[i] = pdf[i] + cdf[i-1]
       
    rcdf=np.zeros(256,dtype=np.uint32)    
    for i in range(0,256):
        rcdf[i] = round(cdf[i]*255)
        new_hist[rcdf[i]] = histogram[i]
    
    equalized_image = np.zeros_like(img)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            equalized_image[i,j] = rcdf [img[i,j]]
    
    return pdf,rcdf,equalized_image,new_hist
    

img=cv2.imread('histogram.jpg',cv2.IMREAD_GRAYSCALE)
shape = int(input("Enter Shape parameter: "))
scale = float(input("Enter Scale parameter: "))

pdf,cdf,equalized_image,mp_func = histogram_equalization(img)

e_distribution = erlang_distribution(shape, scale) 

e_pdf = e_distribution / np.sum(e_distribution)

e_cdf = np.zeros(256,dtype=np.float32)

e_cdf[0]= e_pdf[0]

for i in range(1,256):
    e_cdf[i] =e_pdf[i] + e_cdf[i-1]
#rg_cdf=np.zeros(256,dtype=np.uint32)    
for  i in range(0,256):
    e_cdf[i] = round(e_cdf[i]*255)
plt.subplot(2,3,1)
plt.title("Erlang Distribution")
plt.plot(e_distribution)

plt.subplot(2,3,2)
plt.title("target pdf")
plt.plot(e_pdf)

plt.subplot(2,3,3)
plt.title("target cdf")
plt.plot(e_cdf)
plt.show()

mp = np.zeros(256,dtype = np.float32)
for i in range(256):
    x = (np.abs(e_cdf-cdf[i])).argmin()
    mp[i] = x
output=np.zeros_like(img)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        output[i,j] = mp[img[i,j]]

o_pdf,o_cdf,o_equalized_image,o_mp_func = histogram_equalization(output)

fig, axs = plt.subplots(2, 3, figsize=(15, 10))  

# Plotting the input histogram
axs[0, 0].hist(img.ravel(), 256, [0, 255])
axs[0, 0].set_title("Input Histogram")

# Plotting the PDF of input image
axs[0, 1].plot(pdf)
axs[0, 1].set_title("PDF of Input Image")

# Plotting the CDF of input image
axs[0, 2].plot(cdf)
axs[0, 2].set_title("CDF of Input Image")

# Plotting the output histogram
axs[1, 0].hist(output.ravel(), 256, [0, 255])
axs[1, 0].set_title("Output Histogram")

# Plotting the PDF of output image
axs[1, 1].plot(o_pdf)
axs[1, 1].set_title("PDF of Output Image")

# Plotting the CDF of output image
axs[1, 2].plot(o_cdf)
axs[1, 2].set_title("CDF of Output Image")

# Improving spacing between plots
plt.tight_layout()

# Showing the plots
plt.show()



cv2.imshow("input_img",img)
cv2.imshow("Equalized_img",equalized_image)
cv2.imshow("output_img",output)
cv2.imshow("output_equalized_img",o_equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()