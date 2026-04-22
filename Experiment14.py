import numpy as np
import cv2

# ---------- LOAD IMAGE ----------
img = cv2.imread("image.jpg", 0)
img = cv2.resize(img, (32, 32))
img = img / 255.0  # normalize

# ---------- CONVOLUTION ----------
def convolution(image, kernel):
    k = kernel.shape[0]
    h, w = image.shape
    output = np.zeros((h-k+1, w-k+1))
    
    for i in range(h-k+1):
        for j in range(w-k+1):
            region = image[i:i+k, j:j+k]
            output[i, j] = np.sum(region * kernel)
    
    return output

# Example kernel (edge detector)
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

conv_out = convolution(img, kernel)

# ---------- RELU ----------
relu = np.maximum(0, conv_out)

# ---------- MAX POOLING ----------
def max_pool(image, size=2):
    h, w = image.shape
    output = np.zeros((h//size, w//size))
    
    for i in range(0, h, size):
        for j in range(0, w, size):
            output[i//size, j//size] = np.max(image[i:i+size, j:j+size])
    
    return output

pool = max_pool(relu)

# ---------- FLATTEN ----------
flat = pool.flatten()

# ---------- DENSE LAYER ----------
weights = np.random.rand(len(flat), 2)  # 2 classes
bias = np.random.rand(2)

output = np.dot(flat, weights) + bias

# ---------- SOFTMAX ----------
def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / exp.sum()

prediction = softmax(output)

print("Prediction probabilities:", prediction)
print("Predicted class:", np.argmax(prediction))