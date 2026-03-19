import cv2
import pickle
import csv
from datetime import datetime
import os

# Load face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load trained model
model = cv2.face.LBPHFaceRecognizer_create()
model.read("trainer.yml")

# Load label map
with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

# Create attendance file
file_name = "attendance.csv"

# If file not exists → create with header
if not os.path.exists(file_name):
    with open(file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time"])

# Track marked names
marked_names = set()

cap = cv2.VideoCapture(0)

print("Starting Attendance System...")

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

        if confidence < 70:
            name = labels.get(id_, "Unknown")

            # ✅ Mark attendance only once
            if name not in marked_names:
                marked_names.add(name)

                now = datetime.now()
                time_str = now.strftime("%H:%M:%S")

                with open(file_name, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([name, time_str])

                print(f"{name} marked present at {time_str}")
        else:
            name = "Unknown"

        # Draw rectangle + name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()