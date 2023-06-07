import cv2
import streamlit as st
import numpy as np
import pickle
import mediapipe as mp




def main():
    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)

    # Set video width and height
    
    
    with open(r"G:\Project\Self video Samples\Bicep curl\Bcurl_cvs.pkl", 'rb') as f:
        model = pickle.load(f)
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    

# Define pose classes
    pose_classes = {1: 'Down', 2: 'UP'}

    # Create an empty placeholder to display the video stream
    placeholder = st.empty()
    
    
    while True:
        ret, frame = cap.read()
        with mp_pose.Pose() as pose:
        # Convert BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            try:


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




                landmarks = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
                predicted_class = model.predict([landmarks])[0]
                class_probability = model.predict_proba([landmarks])[0]
                current_class = pose_classes[predicted_class]

    # Render landmarks on the frame
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Read frame from the video capture



        # Display the frame in Streamlit
                if ret:

                    placeholder.image(frame, channels="BGR")
            except:
                pass

    # Break the loop if the 'Esc' key is pressed
            if cv2.waitKey(1) == 27:
                break

# Release the video capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
