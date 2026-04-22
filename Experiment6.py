import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("image.jpg", 0)

if image is None:
    print("Error: Image not found")
    exit()

rows, cols = image.shape
x = np.arange(cols)
y = np.arange(rows)
X, Y = np.meshgrid(x, y)

noise = 30 * np.sin(2 * np.pi * X / 30)
noisy_image = image + noise
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

f = np.fft.fft2(noisy_image)
fshift = np.fft.fftshift(f)

mask = np.ones((rows, cols), np.uint8)

mask[rows//2+10, cols//2+10] = 0
mask[rows//2-10, cols//2-10] = 0

inverse_filtered = fshift * mask
inv_ishift = np.fft.ifftshift(inverse_filtered)
img_inverse = np.fft.ifft2(inv_ishift)
img_inverse = np.abs(img_inverse)


K = 0.01
H = mask 

wiener = (np.conj(H) / (np.abs(H)**2 + K)) * fshift
w_ishift = np.fft.ifftshift(wiener)
img_wiener = np.fft.ifft2(w_ishift)
img_wiener = np.abs(img_wiener)

titles = ['Original', 'Noisy Image', 'Inverse Filter', 'Wiener Filter']
images = [image, noisy_image, img_inverse, img_wiener]

plt.figure(figsize=(10,6))

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()