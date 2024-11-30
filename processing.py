'''
import pandas as pd
import cv2
import numpy as np
import json

# Load your CSV data
df = pd.read_csv("/Users/aashrithalikki/Downloads/xyz/xyz.csv")  # Replace with your actual CSV path
video_urls = df['Video URL'].dropna().unique()  # Get unique video URLs, drop NaN values

# Load YOLO model
def load_yolo_model():
    # Paths to YOLO config, weights, and labels files
    yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = yolo_net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return yolo_net, output_layers, classes

# Detect objects in the video frame
def detect_objects(frame, yolo_net, output_layers):
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
            if confidence > 0.5:  # Threshold for confidence
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-maxima Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    return boxes, confidences, class_ids, indices

# Process the video and detect objects in each frame
def process_video(video_url, yolo_net, output_layers, classes):
    video = cv2.VideoCapture(video_url)
    frame_count = 0
    detected_objects = {}

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_count += 1

        # Detect objects in the current frame
        boxes, confidences, class_ids, indices = detect_objects(frame, yolo_net, output_layers)

        # Check if indices is a tuple and convert to NumPy array if necessary
        if isinstance(indices, tuple):
            if indices:  # Check if the tuple is not empty (has shape (0, 1))
                indices = indices[0]  # Get the NumPy array from the tuple
            else:
                indices = np.array([])  # Create an empty NumPy array if the tuple is empty

        # Count occurrences of detected objects in the video (e.g., logos, influencers)
        for i in indices.flatten():  # Now indices is guaranteed to be a NumPy array
            label = str(classes[class_ids[i]])
            if label not in detected_objects:
                detected_objects[label] = 0
            detected_objects[label] += 1

    video.release()
    return detected_objects, frame_count

# Save results to JSON file after each video
def save_result_to_json(result, json_file_path="/Users/aashrithalikki/Downloads/xyz/res.json"):
    # Load existing results from the JSON file if it exists
    try:
        with open(json_file_path, "r") as f:
            existing_results = json.load(f)
    except FileNotFoundError:
        existing_results = []

    # Append the new result to the existing ones
    existing_results.append(result)

    # Save the updated results back to the JSON file
    with open(json_file_path, "w") as f:
        json.dump(existing_results, f, indent=4)

# Loop through video URLs and process them
def analyze_videos():
    yolo_net, output_layers, classes = load_yolo_model()  # classes are loaded here

    for video_url in video_urls:
        print(f"Processing video: {video_url}")

        # Process each video and get detected objects
        detected_objects, frame_count = process_video(video_url, yolo_net, output_layers, classes)

        # Prepare the result for the current video
        result = {
            'video_url': video_url,
            'detected_objects': detected_objects,
            'frame_count': frame_count
        }

        # Save the result to the JSON file
        save_result_to_json(result)

# Example usage
analyze_videos()

print("Video analysis complete!")
'''
import os
import cv2
import numpy as np
import json
import pandas as pd

# Load your CSV data
df = pd.read_csv("/Users/aashrithalikki/Downloads/xyz/xyz_sorted.csv")  # Replace with your actual CSV path
video_urls = df['Video URL'].dropna().unique()  # Get unique video URLs, drop NaN values

# Load YOLO model
def load_yolo_model():
    # Paths to YOLO config, weights, and labels files
    yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = yolo_net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return yolo_net, output_layers, classes

# Detect objects in the video frame
def detect_objects(frame, yolo_net, output_layers):
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
            if confidence > 0.5:  # Threshold for confidence
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-maxima Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    return boxes, confidences, class_ids, indices

# Process the video and detect objects in each frame
def process_video(video_url, yolo_net, output_layers, classes, output_dir):
    video = cv2.VideoCapture(video_url)
    frame_count = 0
    detected_objects = {}

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_count += 1

        # Detect objects in the current frame
        boxes, confidences, class_ids, indices = detect_objects(frame, yolo_net, output_layers)

        # Check if indices is a tuple and convert to NumPy array if necessary
        if isinstance(indices, tuple):
            if indices:  # Check if the tuple is not empty (has shape (0, 1))
                indices = indices[0]  # Get the NumPy array from the tuple
            else:
                indices = np.array([])  # Create an empty NumPy array if the tuple is empty

        # Count occurrences of detected objects in the video (e.g., logos, influencers)
        for i in indices.flatten():  # Now indices is guaranteed to be a NumPy array
            label = str(classes[class_ids[i]])
            if label not in detected_objects:
                detected_objects[label] = 0
            detected_objects[label] += 1

            # Save the frame if the detected object is a person
            if label == 'person':
                frame_path = os.path.join(output_dir, f"{label}_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame)

    video.release()
    return detected_objects, frame_count

# Save results to JSON file after each video
def save_result_to_json(result, json_file_path="/Users/aashrithalikki/Downloads/xyz/res.json"):
    # Load existing results from the JSON file if it exists
    try:
        with open(json_file_path, "r") as f:
            existing_results = json.load(f)
    except FileNotFoundError:
        existing_results = []

    # Append the new result to the existing ones
    existing_results.append(result)

    # Save the updated results back to the JSON file
    with open(json_file_path, "w") as f:
        json.dump(existing_results, f, indent=4)

# Loop through video URLs and process them
def analyze_videos(output_dir):
    yolo_net, output_layers, classes = load_yolo_model()  # classes are loaded here

    for video_url in video_urls:
        print(f"Processing video: {video_url}")

        # Process each video and get detected objects
        detected_objects, frame_count = process_video(video_url, yolo_net, output_layers, classes, output_dir)

        # Prepare the result for the current video
        result = {
            'video_url': video_url,
            'detected_objects': detected_objects,
            'frame_count': frame_count
        }

        # Save the result to the JSON file
        save_result_to_json(result)

# Create output directory if it doesn't exist
output_dir = "/Users/aashrithalikki/Downloads/xyz/influencer_images"
os.makedirs(output_dir, exist_ok=True)

# Example usage
analyze_videos(output_dir)

print("Video analysis complete!")
