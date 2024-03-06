from ultralytics import YOLO
import cv2
from time import time

def detect_objects_video(video_path, output_path):
    # Load the knife and person detection models
    knife_model = YOLO('./models/knife-best.pt')
    person_model = YOLO('./models/person-best.pt')

    # Open the video file
    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 25  # Set the desired frame rate to 30 fps
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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

        # Set the time for this frame to the current time.
        time2 = time()
        
        # Check if the difference between the previous and this frame time > 0 to avoid division by zero.
        if (time2 - time1) > 0:
            # Calculate the number of frames per second.
            frames_per_second = 1.0 / (time2 - time1)
            
            # Write the calculated number of frames per second on the frame. 
            cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        
        # Update the previous frame time to this frame time.
        time1 = time2
        
        # Write the frame to the output video file.
        output_video.write(frame)
        
        # Display the frame.
        cv2.imshow('Object Detection', frame)
        
        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and writer objects
    video.release()
    output_video.release()
    cv2.destroyAllWindows()

# Provide the path to your video file and where you want to save the output
# detect_objects_video('images/input.mp4', 'results/predict_output.avi')