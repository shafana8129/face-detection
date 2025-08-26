import numpy as np
import cv2 as cv

# Load cascade
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

# Read image
img = cv.imread("face.jpg")
gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray_image, 1.1, 5)

# Draw rectangles & save cropped faces
for i, (x, y, w, h) in enumerate(faces):
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    face_crop = img[y:y+h, x:x+w]
    cv.imwrite(f"facecrop.jpg", face_crop)

# Print detection result
if len(faces) > 1:
    print("Multiple faces detected!!")
elif len(faces) == 1:
    print("Single face detected!!")
else:
    print("No face detected..")

print(faces)

# Show image
cv.imshow("Detected Faces", img)
cv.waitKey(0)
cv.destroyAllWindows()
