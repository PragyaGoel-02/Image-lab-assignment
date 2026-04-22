import cv2
import numpy as np

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Create recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# ---------- TRAINING DATA ----------
faces = []
labels = []

# Load images
img1 = cv2.imread("image.jpg", 0)
img2 = cv2.imread("image2.jpg", 0)

# Check images
if img1 is None or img2 is None:
    print("Error: Training images not found!")
    exit()

# Resize (VERY IMPORTANT)
img1 = cv2.resize(img1, (200, 200))
img2 = cv2.resize(img2, (200, 200))

faces.append(img1)
labels.append(0)

faces.append(img2)
labels.append(1)

# Train model
recognizer.train(faces, np.array(labels))

# ---------- FACE RECOGNITION ----------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces_detected:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))  # match training size
        
        label, confidence = recognizer.predict(face)
        
        if confidence < 100:
            name = "Person 1" if label == 0 else "Person 2"
        else:
            name = "Unknown"
        
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, name, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    cv2.imshow("Face Recognition", frame)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()