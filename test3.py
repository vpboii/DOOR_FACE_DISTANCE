import cv2
import sys
import imutils
import numpy as np
import logging as log
import datetime as dt
from utilities import preprocess_face_frame, decode_prediction, write_bb
from math import pow, sqrt
from imutils.video import FPS
from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model



# TRIED DISTANCE = https://github.com/amolikvivian/Social-Distance-Breach-Detector-OpenCV-DL/blob/master/main.py

FOCAL_LENGTH = 615
WARNING_LABEL = "Maintain Safe Distance. Move away!"

PERSON_ID = 15
CONFIDENCES = 0.4

MODEL_WEIGHTS = 'MobileNetSSD_deploy.caffemodel'
MODEL_CONFIG = 'MobileNetSSD_deploy.prototxt.txt'

#Loading Caffe Model
print('[Status] Loading Model...')
nn = cv2.dnn.readNetFromCaffe(MODEL_CONFIG, MODEL_WEIGHTS)

# Reference : https://github.com/shantnu/Webcam-Face-Detect
# https://github.com/TZebin/pyimagesearch/blob/master/face-mask-detector/detect_mask_video.py

# About cascade : https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
#OpenCV comes with a number of built-in cascades for detecting everything from cases,eyes hand legs etc.

#Loading Mask Model
print('[Status] Loading Model...')
model = load_model("mask_detector.model")
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = VideoStream(src=0).start()
anterior = 0

while True:
    # Capture frame-by-frame
    frame = video_capture.read()
    frame = imutils.resize(frame, width=700)
    #frame1 = cv2.resize(frame, (0, 0), None, 1.5, 1.5)
    
    
    #convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Detect faces in the image 
    faces = faceCascade.detectMultiScale(
        gray, # Turn the image to grayscale
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
   
    
    faces_dict = {"faces_list": [],
                  "faces_rect": []
                  }

                  
    # Draw a rectangle around the faces
    for rect in faces:
        (x, y, w, h) = rect
        face_frame = frame[y:y + h, x:x + w]
        # preprocess image
        face_frame_prepared = preprocess_face_frame(face_frame)
           
      
        faces_dict["faces_list"].append(face_frame_prepared)
        faces_dict["faces_rect"].append(rect)

    if faces_dict["faces_list"]:
        faces_preprocessed = preprocess_input(np.array(faces_dict["faces_list"]))
        preds = model.predict(faces_preprocessed)

        for i, pred in enumerate(preds):
                mask_or_not, confidence = decode_prediction(pred)
                write_bb(mask_or_not, confidence, faces_dict["faces_rect"][i], frame)
        
    #Converting Frame to Blob
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    
    
    #Passing Blob through network to detect and predict
    nn.setInput(blob)
    detections = nn.forward()
        
    #Creating dictionaries to store position and coordinates
    pos = {}
    coordinates = {}
    
    #Loop over the detections
    for i in np.arange(0, detections.shape[2]):

        #Extracting the confidence of predictions
        conf = detections[0, 0, i, 2]

        #Filtering out weak predictions
        if conf > CONFIDENCES:

            #Extracting the index of the labels from the detection
            object_id = int(detections[0, 0, i, 1])

            #Identifying only Person as detected object
            if(object_id == PERSON_ID):
                
                #Storing bounding box dimensions
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype('int')

                #Draw the prediction on the frame
                label = 'Person: {:.1f}%'.format(conf * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (10,255,0), 2)

                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (20,255,0), 1)
                   
                #Adding the bounding box coordinates to dictionary
                coordinates[i] = (startX, startY, endX, endY)

                #Extracting Mid point of bounding box
                midX = abs((startX + endX) / 2)
                midY = abs((startY + endY) / 2)
                
                #Calculating height of bounding box
                ht = abs(endY-startY)

                #Calculating distance from camera
                distance = (FOCAL_LENGTH * 165) / ht
                    
                #Mid-point of bounding boxes in cm
                midX_cm = (midX * distance) / FOCAL_LENGTH
                midY_cm = (midY * distance) / FOCAL_LENGTH
                
                #Appending the mid points of bounding box and distance between detected object and camera 
                pos[i] = (midX_cm, midY_cm, distance)
    
    #Creating list to store objects with lower threshold distance than required
    proximity = []

    #Looping over positions of bounding boxes in frame
    for i in pos.keys():
        for j in pos.keys():
            if i < j:
                
                #Calculating distance between both detected objects
                squaredDist = (pos[i][0] - pos[j][0])**2 + (pos[i][1] - pos[j][1])**2 + (pos[i][2] - pos[j][2])**2
                dist = sqrt(squaredDist)

                #Checking threshold distance - 175 cm and adding warning label
                if dist < 170:
                    proximity.append(i)
                    proximity.append(j)
                    cv2.putText(frame, WARNING_LABEL, (50,50), cv2.FONT_HERSHEY_DUPLEX, 0.5, [0,255,255], 1)
           
           
    for i in pos.keys():
        if i in proximity:
            color = [0,0,255]
        else:
            color = [0,255,0]
        #Drawing rectangle for detected objects
        (x, y, w, h) = coordinates[i]
        cv2.rectangle(frame, (x, y), (w, h), color, 2)
                   
    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

    
    if not i in proximity:
        cv2.imshow('Window 2 - Door Entry Color', frame)
        colored_window = np.zeros((180, 180, 3), dtype='uint8')
        colored_window[:] = 0, 255, 0
        cv2.imshow("Window 2 - Door Entry Color", colored_window)
    else:
        colored_window = np.zeros((180, 180, 3), dtype='uint8')
        colored_window[:] = 0, 0, 255
        cv2.imshow("Window 2 - Door Entry Color", colored_window)

    
    
    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # When everything is done, release the capture
video_capture.stop()
cv2.destroyAllWindows()