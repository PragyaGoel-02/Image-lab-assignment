import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("image.jpg")

if image is None:
    print("Error: Image not found")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower = np.array([35, 50, 50])
upper = np.array([85, 255, 255])

mask = cv2.inRange(hsv, lower, upper)

segmented = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

titles = ['Original', 'HSV Image', 'Mask', 'Segmented Output']
images = [image_rgb, hsv, mask, segmented]

plt.figure(figsize=(10,8))

for i in range(4):
    plt.subplot(2,2,i+1)
    if i == 1:
        plt.imshow(images[i])   
    else:
        plt.imshow(images[i], cmap='gray' if i==2 else None)
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()