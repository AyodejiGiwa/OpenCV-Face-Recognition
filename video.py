import cv2

from main import face_haar_cascade, image

capture = cv2.VideoCapture(0)
face_haar_cascade = cv2(image, cv2.cascadeclassifier('haarcascades'))

while True:
    gray = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)
    faces = face_haar_cascade(gray,1.1,4)

    for (x,y,w,h) in faces:
        cv2.rectangle(capture,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow('Face', face_haar_cascade)
        cv2.waitKey(30) & 0xFF
        if K == 27:
            break

    cv2.release()