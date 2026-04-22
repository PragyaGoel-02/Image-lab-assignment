import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("image.jpg", 0)

if image is None:
    print("Error: Image not found")
    exit()

min_val = np.min(image)
max_val = np.max(image)

contrast_stretched = ((image - min_val) / (max_val - min_val)) * 255
contrast_stretched = contrast_stretched.astype(np.uint8)

hist_eq = cv2.equalizeHist(image)


titles = ['Original', 'Contrast Stretching', 'Histogram Equalization']
images = [image, contrast_stretched, hist_eq]

plt.figure(figsize=(10,5))

for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.show()