import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("image.jpg", 0)

if image is None:
    print("Error: Image not found")
    exit()

f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)

rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

mask_lpf = np.zeros((rows, cols), np.uint8)
mask_lpf[crow-30:crow+30, ccol-30:ccol+30] = 1

lpf = fshift * mask_lpf
lpf_ishift = np.fft.ifftshift(lpf)
img_lpf = np.fft.ifft2(lpf_ishift)
img_lpf = np.abs(img_lpf)

mask_hpf = np.ones((rows, cols), np.uint8)
mask_hpf[crow-30:crow+30, ccol-30:ccol+30] = 0

hpf = fshift * mask_hpf
hpf_ishift = np.fft.ifftshift(hpf)
img_hpf = np.fft.ifft2(hpf_ishift)
img_hpf = np.abs(img_hpf)

titles = ['Original Image', 'Low Pass Filter (Blur)', 'High Pass Filter (Edges)']
images = [image, img_lpf, img_hpf]

plt.figure(figsize=(10,5))

for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()