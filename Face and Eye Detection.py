import cv2
import numpy as np

from google.colab.patches import cv2_imshow

import cv2

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") #Note the change

img = cv2.imread("/content/human.jpeg")
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces=face_cascade.detectMultiScale(gray_img, scaleFactor=1.05,minNeighbors=5)

for x, y, w, h in faces:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

resized=cv2.resize(img,(int(img.shape[1]/3), int(img.shape[0]/3)))

cv2_imshow(resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

for (x,y,w,h) in faces:
 img = cv2.rectangle(img,(x,y),(x+w, y+h),(255,0,0),3)
cv2.imwrite("/content/human.jpeg",img)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml'

eyes = eye_cascade.detectMultiScale(gray_img,1.03, 5)
for (ex,ey,ew,eh) in eyes:
 img = cv2.rectangle(img,(ex,ey),(ex+ew, ey+eh),(0,255,0),2)
resized=cv2.resize(img,(int(img.shape[1]/3), int(img.shape[0]/3)))

cv2_imshow(resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
