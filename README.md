



# **Influencer Detection & Engagement Analysis**

This repository provides a comprehensive solution for detecting influencers in videos using YOLO object detection, analyzing their engagement metrics, and exporting the results to an Excel file for easy analysis and reporting.

---

## **Project Structure**

```plaintext
├── processing.py           # Script to process videos, detect influencers, and save results in JSON
├── final_res.py            # Script to clean, aggregate influencer data, and export to Excel
├── coco.names              # YOLO object detection class names (COCO dataset)
├── res.json                # JSON file storing detected influencers from processed videos
├── xyz.csv                 # CSV file containing video URLs for analysis
├── influencers_summary.xlsx  # Output Excel file with influencer rankings and engagement
```

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/aashritha78/Influencer-Detection-Engagement.git
   cd Influencer-Detection-Engagement
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**

### **1. Process Videos**
Run `processing.py` to detect influencers in the videos listed in `xyz.csv`. The results are saved in `res.json`.

```bash
python processing.py
```

### **2. Aggregate and Export Influencer Data**
Run `final_res.py` to aggregate the detected influencers, calculate average engagement, and export the results to an Excel file.

```bash
python final_res.py
```

---

## **Output**

The final output is saved in `influencers_summary.xlsx`, containing:

- **Influencer Name**: Unique name for each detected influencer.
- **Total Appearances**: Number of times the influencer appeared in the videos.
- **Total Frames**: Total frames where the influencer was detected.
- **Average Engagement**: Engagement score based on appearance per frame.
- **Video URLs**: List of video URLs where the influencer was detected.
- **Rank**: Rank based on average engagement.

---

## **Key Functions**

### **`processing.py`**
- `load_yolo_model()`: Loads the YOLO model, configuration, and COCO class names.
- `detect_objects(frame, yolo_net, output_layers)`: Detects objects in a video frame.
- `process_video(video_url, yolo_net, output_layers, classes)`: Processes each video and detects objects.
- `save_result_to_json(result, json_file_path)`: Saves detected influencer data to a JSON file.

### **`final_res.py`**
- `remove_duplicates(video_data)`: Removes duplicate video entries.
- `aggregate_influencers(video_data)`: Aggregates influencer data across videos.
- `save_to_excel(influencer_summary, file_name)`: Saves the final influencer data to an Excel file.

---

## **Dependencies**

- Python 3.7+
- OpenCV
- NumPy
- Pandas
- YOLOv3 weights and configuration

---

## **Contributing**

Contributions are welcome! Feel free to fork the repository and submit a pull request.

---
