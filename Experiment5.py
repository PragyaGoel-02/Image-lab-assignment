import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("image.jpg", 0)

if image is None:
    print("Error: Image not found")
    exit()


def add_gaussian_noise(img):
    mean = 0
    sigma = 25
    noise = np.random.normal(mean, sigma, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper(img):
    noisy = img.copy()
    prob = 0.02
    
    salt = np.random.rand(*img.shape) < prob
    noisy[salt] = 255
    
    pepper = np.random.rand(*img.shape) < prob
    noisy[pepper] = 0
    
    return noisy

gaussian_noisy = add_gaussian_noise(image)
sp_noisy = add_salt_pepper(image)


mean_filtered = cv2.blur(gaussian_noisy, (5,5))

median_filtered = cv2.medianBlur(sp_noisy, 5)

gaussian_filtered = cv2.GaussianBlur(gaussian_noisy, (5,5), 0)

titles = [
    'Original',
    'Gaussian Noise', 'Mean Filter', 'Gaussian Filter',
    'Salt & Pepper Noise', 'Median Filter'
]

images = [
    image,
    gaussian_noisy, mean_filtered, gaussian_filtered,
    sp_noisy, median_filtered
]

plt.figure(figsize=(12,8))

for i in range(len(images)):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()