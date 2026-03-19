import cv2
import os

# Take user name input
user_name = input("Enter your name: ").strip()

# Create dataset folder with user name
folder = f"dataset/{user_name}"
if not os.path.exists(folder):
    os.makedirs(folder)

# Load face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

count = 0
max_images = 50

print(f"Capturing images for {user_name}...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Save image with numbering
        file_path = os.path.join(folder, f"{user_name}_{count}.jpg")
        cv2.imwrite(file_path, face)

        count += 1

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Face Capture", frame)

    # Stop after 50 images
    if count >= max_images:
        break

    # ESC to exit early
    if cv2.waitKey(50) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"Done! {count} images saved in folder '{folder}'")