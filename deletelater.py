import json
import pandas as pd

# Path to COCO 2014 captions file
json_path = "dataset/data/coco/annotations/captions_val2014.json"

# Load JSON annotations
with open(json_path, "r") as f:
    coco_data = json.load(f)

# Map image_id to filename
image_id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}

# Extract first 5K image-caption pairs
rows = []
for ann in coco_data["annotations"]:
    image_id = ann["image_id"]
    caption = ann["caption"]
    filename = image_id_to_filename.get(image_id)
    if filename:
        image_path = f"dataset/data/coco/val2014/{filename}"
        rows.append({"image_path": image_path, "caption": caption})
    if len(rows) == 5000:
        break

# Save to CSV
df = pd.DataFrame(rows)
df.to_csv("esreal_data.csv", index=False)

print("✅ Saved: esreal_data.csv with 5,000 image-caption pairs")
