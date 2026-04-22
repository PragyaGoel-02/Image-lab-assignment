import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread("image.jpg", 0)

# Convert to binary
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Kernel
kernel = np.ones((3,3), np.uint8)

# Morphological operations
erosion = cv2.erode(binary, kernel, iterations=1)
dilation = cv2.dilate(binary, kernel, iterations=1)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Display
titles = ['Original', 'Binary', 'Erosion', 'Dilation', 'Opening', 'Closing']
images = [image, binary, erosion, dilation, opening, closing]

plt.figure(figsize=(10,8))

for i in range(len(images)):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()