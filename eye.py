import numpy as np
import cv2 as cv

eye_cascade = cv.CascadeClassifier(cv.data.haarcascades+"haarcascade_eye.xml")
img = cv.imread("faceee.jpg")
eyes = eye_cascade.detectMultiScale(img,1.1,5)
for (x,y,w,h) in eyes:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
print(eyes)
cv.imshow("eyes",img)
cv.waitKey(0)