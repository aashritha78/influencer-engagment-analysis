import os
import cv2
import numpy as np
import json
import pandas as pd

# Load your CSV data
df = pd.read_csv("/Users/aashrithalikki/Downloads/xyz/xyz.csv")  # Replace with your actual CSV path
video_urls = df['Video URL'].dropna().unique()  # Get unique video URLs, drop NaN values

# Load YOLO Face Detection Model
def load_yolo_model():
    # Paths to YOLO config, weights, and labels files
    weights_path = "/Users/aashrithalikki/Downloads/xyz/yolov3.weights"
    config_path = "/Users/aashrithalikki/Downloads/xyz/yolov3.cfg"
    yolo_net = cv2.dnn.readNet(weights_path, config_path)
    layer_names = yolo_net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]
    with open("/Users/aashrithalikki/Downloads/xyz/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return yolo_net, output_layers, classes

# Detect faces in the video frame
def detect_faces(frame, yolo_net, output_layers, classes):
    # Prepare the frame for YOLO model
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    detections = yolo_net.forward(output_layers)

    # Process detections
    class_ids, confidences, boxes = [], [], []
    height, width, channels = frame.shape
    for out in detections:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == 'person':  # Threshold for confidence and class filter
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-maxima Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    return boxes, confidences, class_ids, indices

# Process the video and detect faces in each frame
def process_video(video_url, yolo_net, output_layers, classes, output_dir):
    video = cv2.VideoCapture(video_url)
    frame_count = 0
    detected_faces = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_count += 1

        # Detect faces in the current frame
        boxes, confidences, class_ids, indices = detect_faces(frame, yolo_net, output_layers, classes)

        # Check if indices is a tuple and convert to NumPy array if necessary
        if isinstance(indices, tuple):
            if indices:  # Check if the tuple is not empty (has shape (0, 1))
                indices = indices[0]  # Get the NumPy array from the tuple
            else:
                indices = np.array([])  # Create an empty NumPy array if the tuple is empty

        # Count occurrences of detected faces in the video
        for i in indices.flatten():  # Now indices is guaranteed to be a NumPy array
            detected_faces += 1

            # Save the frame if a face is detected
            frame_path = os.path.join(output_dir, f"face_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)

    video.release()
    return detected_faces, frame_count

# Loop through video URLs and process them
def analyze_videos(output_dir):
    yolo_net, output_layers, classes = load_yolo_model()  # Load the YOLO model

    for video_url in video_urls:
        print(f"Processing video: {video_url}")

        # Process each video and get detected faces
        detected_faces, frame_count = process_video(video_url, yolo_net, output_layers, classes, output_dir)

        # Print the result for the current video
        print(f"Video URL: {video_url}")
        print(f"Detected Faces: {detected_faces}")
        print(f"Frame Count: {frame_count}")
        print("-------------------------------------")

# Create output directory if it doesn't exist
output_dir = "/Users/aashrithalikki/Downloads/xyz/influencer_images"
os.makedirs(output_dir, exist_ok=True)

# Example usage
analyze_videos(output_dir)

print("Video analysis complete!")
