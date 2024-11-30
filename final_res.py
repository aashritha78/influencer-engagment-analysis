'''
import json
import pandas as pd
from collections import defaultdict

# Load JSON file with multiple video objects
with open('/Users/aashrithalikki/Downloads/xyz/res.json', 'r') as file:
    video_data = json.load(file)

# Step 1: Remove duplicate video entries (based on JSON content)
def remove_duplicates(video_data):
    # Use a set to store unique JSON objects converted to strings
    unique_videos = list({json.dumps(video, sort_keys=True) for video in video_data})
    # Convert the JSON string back to dict objects
    return [json.loads(video) for video in unique_videos]

# Step 2: Combine influencer data (person counts) across similar JSON structures
def aggregate_influencers(video_data):
    combined_influencers = defaultdict(lambda: {"appearances": 0, "frame_count": 0, "video_urls": []})

    # Iterate through each video entry
    for video in video_data:
        detected_objects = video.get('detected_objects', {})
        person_count = detected_objects.get('person', 0)
        frame_count = video.get('frame_count', 1)
        video_url = video.get('video_url')

        # Only aggregate if there is a 'person' object detected
        if person_count > 0:
            key = f"{person_count}-{frame_count}-{video_url}"  # Unique identifier for the influencer based on count and frame
            combined_influencers[key]["appearances"] += person_count
            combined_influencers[key]["frame_count"] += frame_count
            combined_influencers[key]["video_urls"].append(video_url)

    # Assign unique names to influencers and return summarized data
    influencer_data = [
        {
            "Influencer Name": f"Influencer {idx + 1}",
            "Total Appearances": data["appearances"],
            "Total Frames": data["frame_count"],
            "Average Engagement": round(data["appearances"] / data["frame_count"], 2),
            "Video URLs": ", ".join(data["video_urls"])
        }
        for idx, (influencer, data) in enumerate(combined_influencers.items())
    ]

    # Sort influencers by Average Engagement in descending order and assign Rank
    influencer_data_sorted = sorted(influencer_data, key=lambda x: x["Average Engagement"], reverse=True)
    for rank, influencer in enumerate(influencer_data_sorted, start=1):
        influencer["Rank"] = rank

    return influencer_data_sorted

# Step 3: Save the influencer summary to an Excel file
def save_to_excel(influencer_summary, file_name="influencers_summary.xlsx"):
    df = pd.DataFrame(influencer_summary)
    df.to_excel(file_name, index=False)
    print(f"Influencer data saved to {file_name}")

# Main function to clean, aggregate, and save influencer data to Excel
def main():
    # Remove duplicate video entries
    unique_video_data = remove_duplicates(video_data)

    # Aggregate influencer (person) data from unique videos
    influencer_summary = aggregate_influencers(unique_video_data)

    # Save aggregated data to Excel
    save_to_excel(influencer_summary)

# Run the script
if __name__ == "__main__":
    main()
'''
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Load the influencer summary CSV file
influencer_summary_file_path = "/Users/aashrithalikki/Downloads/xyz/influencers_summary.csv"
df_influencer_summary = pd.read_csv(influencer_summary_file_path)

# Get the top 10 influencers
top_10_influencers = df_influencer_summary.head(10)

# Display the top 10 influencers with their images and metrics
images_dir = "/Users/aashrithalikki/Downloads/xyz/influencer_images"

# Display images and performance in a table format
fig, axes = plt.subplots(nrows=len(top_10_influencers), ncols=3, figsize=(15, 5 * len(top_10_influencers)))

# Get the list of image files in the directory
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])

# Ensure we have at least 10 images
if len(image_files) < 10:
    raise ValueError("Not enough images found in the directory. Ensure there are at least 10 images.")

# Create a mapping from URLs to image files
url_to_image = {}
for image_file in image_files:
    # Assuming the image file name contains the URL or a unique identifier
    url_id = image_file.split('_')[1].split('.')[0]
    if url_id not in url_to_image:
        url_to_image[url_id] = []
    url_to_image[url_id].append(image_file)

# Select images for each rank based on unique URLs
selected_images = []
used_urls = set()

for index, row in top_10_influencers.iterrows():
    video_urls = row['Video URLs'].split(', ')
    for url in video_urls:
        url_id = url.split('/')[-1]  # Extract the unique identifier from the URL
        if url_id in url_to_image and url_id not in used_urls:
            selected_images.append(url_to_image[url_id][0])
            used_urls.add(url_id)
            break

# If not enough unique images are found, repeat images from the same URL
while len(selected_images) < 10:
    for url_id in used_urls:
        if len(selected_images) < 10 and url_to_image[url_id]:
            selected_images.append(url_to_image[url_id].pop(0))

# Ensure we have exactly 10 images
if len(selected_images) < 10:
    raise ValueError("Not enough unique images found for the specified URLs. Ensure there are at least 10 unique images.")

for i, (index, row) in enumerate(top_10_influencers.iterrows()):
    influencer = row['Influencer Name']
    average_performance = row['Average Engagement']
    rank = row['Rank']
    video_urls = row['Video URLs']

    # Load the image for the current rank
    image_path = os.path.join(images_dir, selected_images[i])
    try:
        image = Image.open(image_path)
        axes[i, 0].imshow(image)
        axes[i, 0].axis('off')
    except Exception as e:
        axes[i, 0].text(0.5, 0.5, f'Error: {e}', horizontalalignment='center', verticalalignment='center')
        axes[i, 0].axis('off')

    # Display the rank
    axes[i, 1].text(0.5, 0.5, f"Rank: {rank}", horizontalalignment='center', verticalalignment='center', fontsize=12)
    axes[i, 1].axis('off')

    # Display the average performance
    axes[i, 2].text(0.5, 0.5, f"Average Performance: {average_performance:.2f}", horizontalalignment='center', verticalalignment='center', fontsize=12)
    axes[i, 2].axis('off')

plt.tight_layout()
plt.show()
