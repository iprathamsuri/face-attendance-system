import cv2
import pickle

# Load face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load trained model
model = cv2.face.LBPHFaceRecognizer_create()
model.read("trainer.yml")

# Load label mapping
with open("labels.pkl", "rb") as f:
    label_map = pickle.load(f)

# Reverse map (id → name)
labels = label_map
cap = cv2.VideoCapture(0)

print("Starting Face Recognition...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        id_, confidence = model.predict(face)

        # Convert confidence to readable
        if confidence < 70:
            name = labels[id_]
        else:
            name = "Unknown"

        # Draw rectangle + name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()