# Imports the required libraries 
import cv2
import numpy as np
import time

# Loads the haarcascade files and trains the classifier with those xml files
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

# Sets the camera to the default one
cap = cv2.VideoCapture(0)

# Executes the code all the time to keep capturing while needed
while True:
    # Sets the captured image to a variable
    ret, frame = cap.read()
    # Flips the image in the x axis, otherwise, the image will be inverted
    frame = cv2.flip(frame,1)

    # Transforms the image value into a gray scale and saves it into this variable
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Tries to recognice the faces in the frame, with the trained cascadeClassifier
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w , h) in faces:
        # Draws a rectangle around all faces found
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x: x+w]
        roi_color = frame[y: y+h, x:x+w]

        # Tries to recognice the eyes in the frame, with the trained cascadeClassifier
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew , eh) in eyes:
            # Draws a rectangle around all eyes found
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)


    # Shows the image and sleeps the execution one milisecond
    cv2.imshow('FRec', frame)
    time.sleep(0.1)

    # Breaks the while loop if the q key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Destroys all the window instances generated in this code execution
cv2.destroyAllWindows()
