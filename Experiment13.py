import cv2
import numpy as np

# Load image
image = cv2.imread("image.jpg", 0)

# Convert to binary
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Take first contour
cnt = contours[0]

# Generate simple chain code (8-direction)
chain_code = []

for i in range(len(cnt)-1):
    x1, y1 = cnt[i][0]
    x2, y2 = cnt[i+1][0]
    
    dx = x2 - x1
    dy = y2 - y1
    
    # Map direction
    direction = (dx, dy)
    chain_code.append(direction)

print("Chain Code (first 20):", chain_code[:20])