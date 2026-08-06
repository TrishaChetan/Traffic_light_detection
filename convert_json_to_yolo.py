import json
import os
from PIL import Image

# === CLASS MAPPING ===
class_map = {
    "red": 0,
    "yellow": 1,
    "green": 2
}

json_path = "train.json"
image_dir = "dataset/train_dataset/train_images"
label_dir = "labels/train"
os.makedirs(label_dir, exist_ok=True)

with open(json_path, "r") as f:
    data = json.load(f)

annotations = data.get("annotations", [])

for ann in annotations:
    filename = os.path.basename(ann["filename"]).replace("\\", "/")
    image_path = os.path.join(image_dir, filename)
    label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")

    if not os.path.exists(image_path):
        print(f"⚠️ Image not found: {image_path}")
        continue

    with Image.open(image_path) as img:
        w, h = img.size

    inbox = ann.get("inbox", [])
    for obj in inbox:
        color = obj.get("color")
        box = obj.get("bndbox")
        if color not in class_map or not box:
            continue

        class_id = class_map[color]
        xmin, ymin, xmax, ymax = float(box["xmin"]), float(box["ymin"]), float(box["xmax"]), float(box["ymax"])
        x_center = (xmin + xmax) / 2 / w
        y_center = (ymin + ymax) / 2 / h
        width = (xmax - xmin) / w
        height = (ymax - ymin) / h

        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

        with open(label_path, "a") as out:
            out.write(yolo_line + "\n")

print("✅ YOLO labels created for red/yellow/green lights")
