import cv2
import matplotlib.pyplot as plt

# Load image
image = cv2.imread("image.jpg", 0)

# Global Threshold
_, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Otsu Threshold
_, otsu = cv2.threshold(image, 0, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display
titles = ['Original', 'Global Threshold', 'Otsu Threshold']
images = [image, thresh, otsu]

plt.figure(figsize=(8,4))

for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.show()