import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("image.jpg", 0)

if image is None:
    print("Error: Image not found")
    exit()

edges = cv2.Canny(image, 100, 200)

_, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

titles = ['Original', 'Edges (Canny)', 'Segmented (Threshold)']
images = [image, edges, thresh]

plt.figure(figsize=(10,4))

for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()