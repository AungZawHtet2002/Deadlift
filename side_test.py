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
model = pickle.load(open("both_model.h5","rb"))

def knee_condition(leg,knee,hip):
    leg=np.array(leg)
    knee=np.array(knee)
    hip=np.array(hip)
    radians= np.arctan2(hip[1]-knee[1],hip[0]-knee[0])-np.arctan2(leg[1]-knee[1],leg[0]-knee[0])
    angle=np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle=360-angle    
    return angle


    


with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    while (True):
        cam,frame=cap.read()
        results=pose.process(frame)
        mp_drawing.draw_landmarks(frame,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
        landmarks=results.pose_landmarks.landmark
        print(landmarks)
        
        try:
            keypoints=np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten()
            keypoints=np.delete(keypoints,[0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44])
            X=pd.DataFrame([keypoints])
            label = model.predict(X)[0]
            label= class_name[label]
            right=landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility
            left=landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
            if right>left:
                leg=[landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                knee=[landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                hip=[landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            
                angle=knee_condition(leg,knee,hip)
                if angle<60:
                    output="Knee bent too much"
                else:
                    output="Perfect Form"
            else:
                leg=[landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                knee=[landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                hip=[landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            
                angle=knee_condition(leg,knee,hip)
                if angle<60:
                    output="Knee bent too much"
                else:
                    output="Perfect Form"
            

            if label!= previous:
                counter+=1
                Rep=int(counter/2)
                previous=label
            

            cv2.putText(frame,"Class:",(10,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),1,cv2.LINE_AA)
            cv2.putText(frame,label,(200,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2,cv2.LINE_AA)
            # cv2.putText(frame,"Hand:",(10,100),cv2.FONT_HERSHEY_SIMPLEX,0.8,(150,255,0),1,cv2.LINE_AA)
            cv2.putText(frame,output,(200,100),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)
            # cv2.putText(frame,"Leg:",(10,150),cv2.FONT_HERSHEY_SIMPLEX,0.8,(150,255,0),1,cv2.LINE_AA)
            # cv2.putText(frame,str(angle),(200,120),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2,cv2.LINE_AA)
            cv2.putText(frame,"Rep:",(10,500),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,200),1,cv2.LINE_AA)
            cv2.putText(frame,str(Rep),(200,500),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,200),2,cv2.LINE_AA)
        except Exception as e: 
            pass
        cv2.imshow("Webcam",frame)    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
