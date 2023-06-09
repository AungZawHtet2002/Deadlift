import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd

mp_drawing=mp.solutions.drawing_utils           #To draw landmark point connection on image
mp_pose= mp.solutions.pose                      #To detect landmark points on image
url="http://192.168.100.9:8080/video"
cap = cv2.VideoCapture(url)                      #Using webcam for input image 
previous='up'                                   #Up class is our first class
class_name=['down', 'up']                       #Class names array
counter=0               
Rep=0
model = pickle.load(open("both_model.h5","rb")) #load model file

def knee_condition(leg,knee,hip):
    leg=np.array(leg)
    knee=np.array(knee)
    hip=np.array(hip)
    radians= np.arctan2(hip[1]-knee[1],hip[0]-knee[0])-np.arctan2(leg[1]-knee[1],leg[0]-knee[0])
    angle=np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle=360-angle    
    return angle

# This function is to calculate knee angle to prevent knee injury

    


with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    while (True):
        cam,frame=cap.read()                                                                #Read image from camera
        results=pose.process(frame)                                                         #Processing on image
        mp_drawing.draw_landmarks(frame,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)    #Draw landmarks
        landmarks=results.pose_landmarks.landmark                                           #Landmarks for our next coming steps
        
        
        try:
            keypoints=np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten()       #Flatten to one dimensional array 
            keypoints=np.delete(keypoints,[0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44])    #Delete face landmarks
            X=pd.DataFrame([keypoints])                                                     #Convert np array to data frame
            label = model.predict(X)[0]                                                     #Predict on input data frame 
            label= class_name[label]                                                        #Output class
            right=landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility               #Right knee visibility
            left=landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility                 #Left knee visibility
            if right>left:        #Check left side or right side and if right side 
                leg=[landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]   #Right ankle x,y values
                knee=[landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]    #Right knee x,y values
                hip=[landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]       #Right hip x,y values
            
                angle=knee_condition(leg,knee,hip)
                if angle<60:    #Defining threshold angle
                    output="Knee bent too much"         #Bending knee less than 60' can cause knee, ankle injury 
                else:
                    output="Perfect Form"
            else:   #If left side
                leg=[landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y] #Left ankle x,y values
                knee=[landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]  #Left knee x,y values
                hip=[landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]     #Left hip x,y values
            
            
                angle=knee_condition(leg,knee,hip)
                if angle<60:    #Defining threshold angle
                    output="Knee bent too much"          #Bending knee less than 60' can cause knee, ankle injury 
                else:
                    output="Perfect Form"
            

            if label!= previous:
                counter+=1
                Rep=int(counter/2)      #To count how many time we do in correct form
                previous=label
            

            cv2.putText(frame,"Position:",(10,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),1,cv2.LINE_AA)   #To put text on image
            cv2.putText(frame,label,(200,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2,cv2.LINE_AA)        #To put output result on image
            
            cv2.putText(frame,output,(200,100),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)    #To put text when we are doing right or wrong form
            
            cv2.putText(frame,"Rep:",(10,500),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,200),1,cv2.LINE_AA)     #To put repetation how many we do
            cv2.putText(frame,str(Rep),(200,500),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,200),2,cv2.LINE_AA)  #To put repetation how many we do
        except Exception as e: 
            pass
        cv2.imshow("Webcam",frame)              #Show real time video output
        if cv2.waitKey(1) & 0xFF == ord('q'):   #Break if you press q on keyboard
            break

cap.release()
cv2.destroyAllWindows()
