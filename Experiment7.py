import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("image.jpg")

if image is None:
    print("Error: Image not found")
    exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

rows, cols = image.shape[:2]

tx, ty = 100, 50   

translation_matrix = np.float32([
    [1, 0, tx],
    [0, 1, ty]
])

translated = cv2.warpAffine(image, translation_matrix, (cols, rows))

scaled = cv2.resize(image, None, fx=1.5, fy=1.5)

center = (cols // 2, rows // 2)
angle = 45

rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
rotated = cv2.warpAffine(image, rotation_matrix, (cols, rows))

titles = ['Original', 'Translated', 'Scaled', 'Rotated']
images = [image, translated, scaled, rotated]

plt.figure(figsize=(10,8))

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()