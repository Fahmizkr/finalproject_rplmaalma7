import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to perform object detection
def run_object_detection(image_np, model):
    image_tensor = tf.convert_to_tensor(image_np)
    image_tensor = image_tensor[tf.newaxis, ...]
    output = model(image_tensor)
    return output

# Function to draw bounding boxes
def draw_bounding_boxes(image_np, detection_output, threshold=0.5):
    height, width, _ = image_np.shape
    for i in range(len(detection_output['detection_boxes'][0])):
        box = detection_output['detection_boxes'][0][i].numpy()
        score = detection_output['detection_scores'][0][i].numpy()
        class_id = detection_output['detection_classes'][0][i].numpy()

        if score > threshold:
            ymin, xmin, ymax, xmax = box
            start_point = (int(xmin * width), int(ymin * height))
            end_point = (int(xmax * width), int(ymax * height))
            color = (0, 255, 0)
            thickness = 2
            image_np = cv2.rectangle(image_np, start_point, end_point, color, thickness)
            label = f"Class: {int(class_id)}, Score: {score:.2f}"
            cv2.putText(image_np, label, (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    return image_np

# Capture video from camera
cap = cv2.VideoCapture(1) # 1 for webcam, 0 for internal camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    detection_output = run_object_detection(frame, model, 0.3)

    # Draw bounding boxes
    frame = draw_bounding_boxes(frame, detection_output)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()