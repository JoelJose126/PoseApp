import numpy as np
import mediapipe as mp
import cv2
import time
import pickle
from playsound import playsound
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
import numpy as np
counter = 0
fdl=' '
fdr=' '
a=0
with open(r"G:\Project\Self video Samples\Bicep curl\/Bcurl_cvs.pkl", 'rb') as f:
    model = pickle.load(f)

# Define the classes for pose classification
pose_classes = {1: 'Down', 2: 'UP',}
from pygame import mixer 
mixer.init()
def vc(x):
    
# Starting the mixer
   
    a=0
    # Loading the song
    lsh=r"C:\Users\HP\Desktop\Custom_Det\Left_Shoulder_Wide.wav"
    rsh=r"C:\Users\HP\Desktop\Custom_Det\Right_Shoulder_wide.wav"
    lsl=r"C:\Users\HP\Desktop\Custom_Det\Left_Shoulder_Low.wav" 
    rsl=r"C:\Users\HP\Desktop\Custom_Det\Right_Shoulder_Low.wav"
    
    if x == 1 :
        s=lsh
    elif x==2:
        s=rsh
    if x==3:
        s=lsl
    elif x==4:
        s=rsl
    
        
    # Setting the volume
    
    one=mixer.Sound(s)
    for i in range(0,1000):
        a=a+1
    a=0 
    one.play()
    
    
    
      



def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

def vf():
    counter = 0
    cap = cv2.VideoCapture(0)
    shoulder1,shoulder2,elbow1,elbow2,hip1,hip2=0,0,0,0,0,0
    
# Curl counter variables

    stage = None

## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
        
        # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            image.flags.writeable = False
      
        # Make detection
            results = pose.process(image)
    
        # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                landmarks = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()

        # Perform pose classification using the pickle model
                predicted_class = model.predict([landmarks])[0]
                class_probability = model.predict_proba([landmarks])[0]
                current_class = pose_classes[predicted_class]
                if (class_probability[0] < class_probability[1] < .7) or (class_probability[1] < class_probability[0] < .4):
                    current_class = 'Half'

        # Display the predicted class and probability on the frame
                label = current_class
                prob = class_probability
                cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(image, str(prob), (190, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
                shoulder1 = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                shoulder2 = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow1 = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist1 = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                elbow2 = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist2 = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                hip1=[landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                hip2=[landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            # Calculate angle
                angle1 = calculate_angle(shoulder1, elbow1, wrist1)
                angle2= calculate_angle(shoulder2, elbow2, wrist2)
                angle3 = calculate_angle( elbow1,shoulder1, hip1)
                angle4 = calculate_angle( elbow2,shoulder2, hip2)
           
            
            
            # Visualize angle
                cv2.putText(image, str(angle1), 
                           tuple(np.multiply(elbow1, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
                cv2.putText(image, str(angle2), 
                           tuple(np.multiply(elbow2, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
                cv2.putText(image, str(angle3), 
                           tuple(np.multiply(shoulder1, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
                cv2.putText(image, str(angle4), 
                           tuple(np.multiply(shoulder2, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )

#Exchange angles for flipped image

#             if angle3 > 25 and angle4 > 25  :
#                 stage="B HIGH"
#                 cv2.putText(image, stage, 
#                     (60,60), 
#                     FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
            
            
#             elif angle4 > 25  :
#                 stage=" HIGH"
#                 cv2.putText(image, stage, 
#                     (130,200), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
#             elif angle3 > 25  :
#                 stage="RIGHT HIGH"
#                 cv2.putText(image, stage, 
#                     (60,60), 
#                     FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
#             if angle3 < 25 and angle4 < 25 :
#                 stage="GOOD"
#                 cv2.putText(image, stage, 
#                     (60,60), 
#                     FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
 #             if angle4 < 10 :
#                  cv2.putText(image, 'right LOW' , (15,12), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                
  
 #             if angle3 > 30 :
#                  cv2.putText(image, 'LEFT HIGH' , (15,12), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                
#             if angle3 > 30 :
#                 cv2.putText(image, 'RIGHT HIGH' , (15,12), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    


            # Curl counter logic
#                 if angle1 > 170 and angle2 > 170 and angle3>=8 and angle3<= 15 and angle4>=8 and angle4<= 15 :
#                     stage = "down"
#                 if angle1 <30 and angle2 < 30 and stage =='down':
#                     stage="up"
#                     counter +=1
#                   print(counter)
                
                       
            except:
                pass
        
        # Render curl counter
        
        # Setup status box
            
            cv2.rectangle(image, (0,200), (120,250), (245,117,16), -1)
            cv2.rectangle(image, (0,120), (0,200), (0,0,0), -1)
            cv2.rectangle(image, (520,200), (640,250), (0,255,0), -1)
        
            h,w,c=image.shape
            s1x,s2x=int(round(shoulder1[0]*w)),int(round(shoulder2[0]*w))
            s1y,s2y=int(round(shoulder1[1]*h)),int(round(shoulder2[1]*h))
            e1x,e2x=int(round(elbow1[0]*w)),int(round(elbow2[0]*w))
            e1y,e2y=int(round(elbow1[1]*h)),int(round(elbow2[1]*h))
            cv2.line(image, (s1x,s1y), (s2x,s2y), (0,255,0), 9)
            cv2.circle(image, (s1x,s1y), 7, (255,0,0), -1)
            cv2.circle(image, (s2x,s2y), 7, (255,0,0), -1)
        

            blue = (255, 127, 0)
            red = (50, 50, 255)
            green = (127, 255, 0)
            dark_blue = (127, 20, 0)
            light_green = (127, 233, 100)
            yellow = (0, 255, 255)
            pink = (255, 0, 255)
        # Stage data
#         cv2.putText(image, 'STAGE', (65,12), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            if angle4 > 25  :
                vc(1)
                fdl='Wide'
                
            elif angle4 < 7 :
                vc(3) 
                fdl='Narrow'
            elif angle4 <25 and angle4 >7 :
                fdl='Good'
                
            if angle3 > 25  :
                vc(2)
                fdr='Wide'
            elif angle3 < 7 :
                vc(4)
                fdr='Narrow'
            elif angle3 <25 and angle3 >7 :
                fdr='Good'    
                
                
            if fdl=='Wide'  :
               
                cv2.putText(image,'Wide', 
                        (5,230), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                cv2.line(image, (s2x,s2y), (e2x,e2y), (0,0,255), 9)
               
            
            elif fdl=='Narrow' :
                cv2.putText(image,'Narrow', 
                    (5,230), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                cv2.line(image, (s2x,s2y), (e2x,e2y), (255,0,0), 9)
#             cv2.line(image, (s2,s1), (e2,e1), (0,0,255), 9) 
                
            elif fdl=='Good' :
                
                cv2.putText(image,'GOOD', 
                    (5,230), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                cv2.line(image, (s2x,s2y), (e2x,e2y), (0,255,0), 9)
                
            if fdr=='Wide'  :
                
                cv2.putText(image,'Wide', 
                    (525,230), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                cv2.line(image, (s1x,s1y), (e1x,e1y), (0,0,255), 9)
               
            elif fdr=='Narrow' :
                
                cv2.putText(image,'Narrow', 
                    (525,230), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                cv2.line(image, (s1x,s1y), (e1x,e1y), (255,0,0), 9)
                
            elif fdr=='Good' :
                
                cv2.putText(image,'GOOD', 
                    (525,230), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                cv2.line(image, (s1x,s1y), (e1x,e1y), (0,255,0), 9)
            
            if angle1 > 170 and angle2 > 170 and angle3>=8 and angle3<= 15 and angle4>=8 and angle4<= 15 :
                stage = "down"
            if angle1 <30 and angle2 < 30 and stage =='down'and fdr=='Good' and fdl=='Good':
                    stage="up"
                    counter +=1# Rep data
            
            # Rep data
            cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2, cv2.LINE_AA)
            
                    
            cv2.circle(image, (s1x,s1y), 7, (255,0,0), -1)
            cv2.circle(image, (s2x,s2y), 7, (255,0,0), -1)
            
           
        
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=red, thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=yellow, thickness=1, circle_radius=2)
                                 )
       
       
            cv2.imshow('Curl', image)
        
        
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                exit(0)
                
        cap.release()
        cv2.destroyAllWindows()
    
vf()