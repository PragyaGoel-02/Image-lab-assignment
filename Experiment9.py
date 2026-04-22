import cv2
from collections import Counter
import heapq

image = cv2.imread("image.jpg", 0)

pixels = image.flatten()

freq = Counter(pixels)

class Node:
    def __init__(self, val, freq):
        self.val = val
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

heap = [Node(k, v) for k, v in freq.items()]
heapq.heapify(heap)

while len(heap) > 1:
    left = heapq.heappop(heap)
    right = heapq.heappop(heap)
    merged = Node(None, left.freq + right.freq)
    merged.left = left
    merged.right = right
    heapq.heappush(heap, merged)

codes = {}
def generate(node, code=""):
    if node:
        if node.val is not None:
            codes[node.val] = code
        generate(node.left, code + "0")
        generate(node.right, code + "1")

generate(heap[0])

print("Sample Huffman Codes:", list(codes.items())[:5])