import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r'images\cameraman.tif', cv2.IMREAD_GRAYSCALE)
img_ = img.copy()

# the lower threshold value
T1 = 100

# the upper threshold value
T2 = 180

### create an array of zeros
img_new = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        # print(img[i,j])
        if T1 < img[i,j] and img[i,j] < T2:
            img_new[i,j] = 225
        else:
            img_new[i,j] = 25  # img[i,j]

x = np.arange(0, 256, 1)
x1 = np.array(x)
print(x)

y3 = np.zeros_like(x1)
for i in range(0, len(x1)):
    if(x1[i] < T2 and x1[i] > T1):
        y3[i] = 225
    else:
        y3[i] = 25  # x1[i]
