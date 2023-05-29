import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd

mp_drawing=mp.solutions.drawing_utils
mp_pose= mp.solutions.pose
cap = cv2.VideoCapture(0)
previous='up'
class_name=['down', 'up']
counter=0
Rep=0
model = pickle.load(open("front_model.h5","rb"))

def hand_position(left,right):
    hand_dec="OK"
    leg_dec="OK"
    hand_position=left[0]-right[0]
    leg_position=left[1]-right[1]
    if hand_position < 0.07:
        hand_dec="Narrow grid"
    elif hand_position > 0.15:
        hand_dec="Wide grid"
    if leg_position < 0.04:
        leg_dec="Narrow leg"
    elif leg_position > 0.15:
        leg_dec="Wide leg"
    
    return hand_dec,leg_dec
    


with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    while (True):
        cam,frame=cap.read()
        results=pose.process(frame)
        mp_drawing.draw_landmarks(frame,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
        landmarks=results.pose_landmarks.landmark
        left_position=[landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x]
        right_position=[landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x]
        
        try:
            keypoints=np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten()
            keypoints=np.delete(keypoints,[0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44])
            X=pd.DataFrame([keypoints])
            label = model.predict(X)[0]
            label= class_name[label]
            hand_dec,leg_dec=hand_position(left_position,right_position)
        
            if label!= previous:
                counter+=1
                Rep=int(counter/2)
                previous=label
            #print(Rep)

            cv2.putText(frame,"Class:",(10,50),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),1,cv2.LINE_AA)
            cv2.putText(frame,label,(200,50),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),2,cv2.LINE_AA)
            cv2.putText(frame,"Hand:",(10,100),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),1,cv2.LINE_AA)
            cv2.putText(frame,hand_dec,(200,100),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)
            cv2.putText(frame,"Leg:",(10,150),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),1,cv2.LINE_AA)
            cv2.putText(frame,leg_dec,(200,150),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)
            cv2.putText(frame,"Rep:",(10,500),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,100),1,cv2.LINE_AA)
            cv2.putText(frame,str(Rep),(200,500),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,100),2,cv2.LINE_AA)
        except Exception as e: 
            pass
        cv2.imshow("Webcam",frame)    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
