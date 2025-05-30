import os
import time
import csv
from datetime import datetime

import requests
from ultralytics import RTDETR

# URL of the image
url = "https://trafficcam.calgary.ca/loc37.jpg"   # A test image URL from Calgary's traffic camera network

# Folder to save images and CSV
folder = "loc37"
os.makedirs(folder, exist_ok=True)

# Path to the CSV file
csv_path = os.path.join(folder, "vehicle_counts4.csv")
# Initialize CSV with header if it doesn't exist
if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "person", "car", "motorcycle", "bus", "train", "truck"])

# Load YOLOv8 model
model = RTDETR("rtdetr-l.pt")

# COCO class IDs → names for vehicles
CLASS_MAP = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    6: "train",
    7: "truck"
}

def download_and_count():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_filename = f"{timestamp}.jpg"
    img_path = os.path.join(folder, img_filename)

    try:
        # 1. Download image
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        with open(img_path, "wb") as f:
            f.write(resp.content)

        # 2. Run YOLOv8 inference
        results = model(img_path)[0]
        class_ids = results.boxes.cls.cpu().numpy().astype(int).tolist()

        # 3. Count by class
        counts = {name: 0 for name in CLASS_MAP.values()}
        for cls in class_ids:
            if cls in CLASS_MAP:
                counts[CLASS_MAP[cls]] += 1

        # 4. Append to CSV
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                timestamp,
                counts["person"], 
                counts["car"],
                counts["motorcycle"],
                counts["bus"],
                counts["train"],
                counts["truck"],
            ])

        print(f"[{timestamp}] Counts: {counts}")

    except Exception as e:
        print(f"[{timestamp}] Error: {e}")

    finally:
        # 5. Clean up image file
        if os.path.exists(img_path):
            os.remove(img_path)

if __name__ == "__main__":
    while True:
        download_and_count()
        time.sleep(300)  # wait 5 minutes
