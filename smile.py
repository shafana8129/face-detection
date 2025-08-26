import numpy as np
import cv2 as cv

smile_cascade = cv.CascadeClassifier(cv.data.haarcascades+"haarcascade_smile.xml")
img = cv.imread("smile.jpg")
smile = smile_cascade.detectMultiScale(img,1.1,5)
for (x,y,w,h) in smile:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
print(smile)
cv.imshow("eyes",img)
cv.waitKey(0)