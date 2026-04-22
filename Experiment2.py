import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("image.jpg", 0)

def sampling(img, scale):
    height, width = img.shape
    new_size = (int(width * scale), int(height * scale))
    
    sampled = cv2.resize(img, new_size, interpolation=cv2.INTER_NEAREST)
    
    restored = cv2.resize(sampled, (width, height), interpolation=cv2.INTER_NEAREST)
    return restored

def quantization(img, levels):
    # Normalize and quantize
    quantized = np.floor(img / (256 / levels)) * (256 / levels)
    return quantized.astype(np.uint8)

sampling_rates = [1, 0.5, 0.25]
sampled_images = [sampling(image, s) for s in sampling_rates]

quant_levels = [256, 64, 16, 4]
quantized_images = [quantization(image, q) for q in quant_levels]

plt.figure(figsize=(10, 8))

plt.subplot(3, 4, 1)
plt.imshow(image, cmap='gray')
plt.title("Original")
plt.axis('off')

for i, img in enumerate(sampled_images):
    plt.subplot(3, 4, i+2)
    plt.imshow(img, cmap='gray')
    plt.title(f"Sampling {sampling_rates[i]}")
    plt.axis('off')

for i, img in enumerate(quantized_images):
    plt.subplot(3, 4, i+6)
    plt.imshow(img, cmap='gray')
    plt.title(f"Quant {quant_levels[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()