import cv2
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access camera")
    exit()

ret, frame = cap.read()

if ret:
    cv2.imshow("Captured Image", frame)
    
    cv2.imwrite("captured_image.jpg", frame)
    
    cv2.waitKey(0)
else:
    print("Failed to capture image")

cap.release()
cv2.destroyAllWindows()