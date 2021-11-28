import cv2
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
# Read the image
image = cv2.imread("C:/Users/Nestor/Pictures/Camera Roll/Nestor.jpg") #--imread() helps in loading an image into jupyter including its pixel values
print(image.shape)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Convert image to grayscale. The second argument in the following step is cv2.COLOR_BGR2GRAY, which converts colour
# image to grayscale.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("Original Image:")
# plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
# as opencv loads in BGR format by default, we want to show it in RGB.
# plt.show()
gray.shape
# flatten array of pixels
data = np.array(gray)
flattened = data.flatten()
print(flattened.shape)
# 3x3 array for edge detection
mat_y = np.array([[-1, -2, -1],
                  [0, 0, 0],
                  [1, 2, 1]])
mat_x = np.array([[-1, 0, 1],
                  [0, 0, 0],
                  [1, 2, 1]])

filtered_image = cv2.filter2D(gray, -1, mat_y)
plt.imshow(filtered_image, cmap='gray')
plt.show()
filtered_image = cv2.filter2D(gray, -1, mat_x)
plt.imshow(filtered_image, cmap='gray')
plt.show()
