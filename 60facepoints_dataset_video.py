#================================================
# 
# 60facepoints_dataset_video.py
# 
# Script that captures frames from the computer's webcam, tries to detect a face
# in the frame and then saves the normalized distance between the center of the
# face detected and 60 facepoints in a Dataframe that can be saved as a dataset.
#
#
# Written by: Matheus Inoue, 2018, Federal University of ABC (UFABC)
#=================================================

import cv2
import numpy as np
import dlib  
import pandas as pd
import math
from time import time


# Initialization
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

dfRow = pd.DataFrame(columns = range(0,60))
count = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')  


# Start capture
while count < 500:
    start = time()

    ret, frameOrig = cap.read()
    height = frameOrig.shape[0]
    width = frameOrig.shape[1]
    
    # resize the captured frame to half of its size
    frame = cv2.resize(frameOrig, (int(width/2), int(height/2))) 
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    # Uncomment the following line to check the grayscale picture captured
    # cv2.imshow("Gray", gray)  

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    claheImg = clahe.apply(gray)
    # Uncomment the following line to check the picture after preprocessing
    # cv2.imshow("CLAHE", claheImg)
     
     
    faces = detector(claheImg, 0)  
    print("Found {0} faces!".format(len(faces)))  
      

    if len(faces) >= 1:
        maxArea = 0
        for rect in faces: 
            print(rect.area())
            if rect.area() >= maxArea:
                rectFace = rect
                maxArea = rect.area()

        
          
        landmarks = np.matrix([[p.x, p.y]  
                      for p in predictor(frame, rectFace).parts()])  
          
        landmarks_display = landmarks[:60]  
        x, y, w, h = cv2.boundingRect(landmarks_display)
          
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
          
        centerX = int(x + round(w/2))
        centerY = int(y + round(h/2))
        cv2.circle(frame,(centerX,centerY), 2, (0,0,255), 2)
          
        for idx, point in enumerate(landmarks_display):
            pos = (point[0, 0], point[0, 1])  
            cv2.circle(frame, pos, 2, color=(0, 255, 255), thickness=-1)  
        
        d = []
        for ppoints in landmarks_display:
            d.append(math.sqrt((centerX-ppoints[0,0])**2+(centerY-ppoints[0,1])**2))
        
        dnp = np.array(d)
        dd = dnp/max(dnp)    
        
        dfRow.loc[count] = dd
        count = count + 1
        
    cv2.putText(frame,'Samples: '+str(count),(10,30), font, 0.8,(0,0,255),2)    
    cv2.imshow("Frame captured", frame) 
    
    stop = time()
    print(1/(stop-start))
    if cv2.waitKey(1) & 0xFF == ord ('q'):
        break
    

cv2.destroyAllWindows()
cap.release()


# Uncomment the following line to save a csv file with the data captured
dfRow.to_csv('datasets/face_center2.csv', index=False)
