from ultralytics import YOLO
import cv2
from time import time
import mediapipe as mp
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detectPose(image, pose, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image, 
                and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                    connections=mp_pose.POSE_CONNECTIONS)
        
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
    
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
    
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        
    # Otherwise
    else:
        
        # Return the output image and the found landmarks.
        return output_image, landmarks

def main(input_video_path, output_video_path):
    # Load the knife and person detection models
    knife_model = YOLO('./models/knife-best.pt')
    person_model = YOLO('./models/person-best.pt')

    # Setup Pose function for video.
    mp_pose = mp.solutions.pose
    pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
    mp_drawing = mp.solutions.drawing_utils
    
    # Open the video file
    video = cv2.VideoCapture(input_video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Initialize a variable to store the time of the previous frame.
    time1 = 0

    # Iterate until the video is accessed successfully.
    while video.isOpened():
        # Read a frame.
        ok, frame = video.read()
        
        # Check if frame is not read properly.
        if not ok:
            break
        
        # Perform object detection on the frame
        knife_boxes = knife_model.predict(frame)
        person_boxes = person_model.predict(frame)

        # Draw bounding boxes for knives
        for result in knife_boxes:
            if len(result.boxes.xyxy) > 0:
                cv2.rectangle(frame, (int(result.boxes.xyxy[0, 0].item()), int(result.boxes.xyxy[0, 1].item())), (int(result.boxes.xyxy[0, 2].item()), int(result.boxes.xyxy[0, 3].item())), (0, 0, 255), 2)
                cv2.putText(frame, f'Knife->{result.boxes.conf.item()}', (int(result.boxes.xyxy[0, 0].item()), int(result.boxes.xyxy[0, 1].item()) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw bounding boxes for persons
        for result in person_boxes:
            if len(result.boxes.xyxy) > 0:
                conf = result.boxes.conf[0].item()  # Extract the confidence value
                cv2.rectangle(frame, (int(result.boxes.xyxy[0, 0].item()), int(result.boxes.xyxy[0, 1].item())), (int(result.boxes.xyxy[0, 2].item()), int(result.boxes.xyxy[0, 3].item())), (0, 0, 255), 2)
                cv2.putText(frame, f'Person->{conf}', (int(result.boxes.xyxy[0, 0].item()), int(result.boxes.xyxy[0, 1].item()) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Perform Pose landmark detection.
        frame, landmarks = detectPose(frame, pose_video, display=False)
        
        # Classify Pose and add angle information
        if landmarks:
            frame, _ = classifyPose(landmarks, frame, display=False)
        
        # Set the time for this frame to the current time.
        time2 = time()
        
        # Check if the difference between the previous and this frame time > 0 to avoid division by zero.
        if (time2 - time1) > 0:
            # Calculate the number of frames per second.
            frames_per_second = 1.0 / (time2 - time1)
            
            # Write the calculated number of frames per second on the frame. 
            cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        
        # Update the previous frame time to this frame time.
        time1 = time2
        
        # Write the frame to the output video file.
        output_video.write(frame)
        
        # Display the frame.
        cv2.imshow('Object and Pose Detection', frame)
        
        # Wait until a key is pressed.
        # Retrieve the ASCII code of the key pressed
        k = cv2.waitKey(1) & 0xFF
        
        # Check if 'ESC' is pressed.
        if(k == 27):
            break

    # Release the VideoCapture and VideoWriter objects.
    video.release()
    output_video.release()
    cv2.destroyAllWindows()

def classifyPose(landmarks, output_image, display=False):
    '''
    This function classifies yoga poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''
    mp_pose = mp.solutions.pose

    # Get the angle between the left elbow, shoulder and hip points. 
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    # Get the angle between the right hip, shoulder and elbow points. 
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value])

    print("Right Shoulder Angle ->", right_shoulder_angle )
    print("Left Shoulder Angle ->", left_shoulder_angle )
    
    label = f"Angle->{right_shoulder_angle} & {left_shoulder_angle}"
    cv2.putText(output_image, label, (10, 330), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0), 2 )
    
    # Check if the resultant image is specified to be displayed.
    if display:
    
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    else:
        
        # Return the output image and the classified label.
        return output_image, label

def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.

    '''

    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle is less than zero.
    if angle < 0:

        # Add 360 to the found angle.
        angle += 360
    
    # Return the calculated angle.
    return angle


# Example usage:
main('images/input1.mp4', 'results/combined_output.avi')
