from ultralytics import YOLO
import cv2

def detect_objects(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Use the knife detection model
    knife_model = YOLO('models\knife-best.pt')
    knife_boxes = knife_model.predict(image)
    # print(knife_boxes)
    
    # for result in knife_boxes:
    # # detection
    #     print(result.boxes.xyxy[0, 0].item())
    #     print(result.boxes.xyxy[0, 1].item()) 
    #     print(result.boxes.xyxy[0, 2].item()) 
    #     print(result.boxes.xyxy[0, 3].item()) # box with xyxy format, (N, 4) 
    #     print(result.boxes.conf)   # confidence score, (N, 1)
    #     print(result.boxes.cls)    # cls, (N, 1)

        # # segmentation
        # print(result.masks.masks)     # masks, (N, H, W)
        # print(result.masks.segments)  # bounding coordinates of masks, List[segment] * N

        # # classification
        # print(result.probs)     # cls prob, (num_class, )

    # Use the person detection model
    person_model = YOLO('models\person-best.pt')
    person_boxes = person_model.predict(image)
    # print(person_boxes)

    # Draw bounding boxes for knives
    for result in knife_boxes:
        if len(result.boxes.xyxy) > 0:
            cv2.rectangle(image, (int(result.boxes.xyxy[0, 0].item()), int(result.boxes.xyxy[0, 1].item())), (int(result.boxes.xyxy[0, 2].item()), int(result.boxes.xyxy[0, 3].item())), (0, 0, 255), 2)
            cv2.putText(image, f'Knife->{result.boxes.conf.item()}', (int(result.boxes.xyxy[0, 0].item()), int(result.boxes.xyxy[0, 1].item()) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # # Draw bounding boxes for persons
    for result in person_boxes:
        if len(result.boxes.xyxy) > 0:
            cv2.rectangle(image, (int(result.boxes.xyxy[0, 0].item()), int(result.boxes.xyxy[0, 1].item())), (int(result.boxes.xyxy[0, 2].item()), int(result.boxes.xyxy[0, 3].item())), (0, 0, 255), 2)
            cv2.putText(image, f'Person->{result.boxes.conf[0]}', (int(result.boxes.xyxy[0, 0].item()), int(result.boxes.xyxy[0, 1].item()) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # print(result.boxes.xyxy)

    # Display the result
    # cv2.imshow('Object Detection', image)
    cv2.imwrite('results\Object Detection1.jpg', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



detect_objects('images\images1.jpeg')


