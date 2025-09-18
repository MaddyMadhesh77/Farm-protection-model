import os
import cv2
import shutil

input_root = r"C:\Users\Dell\Downloads\animal-ds\Farm-Harmful-Animal-Dataset"
output_root = "FarmHarmfulAnimalYOLO"

# Create output structure
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_root, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_root, "labels", split), exist_ok=True)

# Class list (from train folder)
class_names = sorted([d for d in os.listdir(os.path.join(input_root, "train")) if os.path.isdir(os.path.join(input_root, "train", d))])
class_to_id = {cls: i for i, cls in enumerate(class_names)}

# Split mapping
split_map = {"train": "train", "validation": "val", "test": "test"}

for split in ["train", "validation", "test"]:
    split_path = os.path.join(input_root, split)
    out_split = split_map[split]

    for cls in os.listdir(split_path):
        cls_path = os.path.join(split_path, cls)
        label_dir = os.path.join(cls_path, "Label")
        if not os.path.isdir(cls_path):
            continue

        # Copy images
        for file in os.listdir(cls_path):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                src = os.path.join(cls_path, file)
                dst = os.path.join(output_root, "images", out_split, file)
                shutil.copy(src, dst)

        # Process labels
        if not os.path.exists(label_dir):
            continue

        for lbl_file in os.listdir(label_dir):
            if not lbl_file.endswith(".txt"):
                continue

            with open(os.path.join(label_dir, lbl_file), "r") as f:
                raw_line = f.read().strip()

            try:
                coords_str = raw_line.split("[")[1].split("]")[0]
                coords = [float(x) for x in coords_str.split()]
            except:
                continue

            if len(coords) != 3:
                continue

            x_center, y_center, w_box = coords
            h_box = w_box  # square assumption

            # Find matching image
            img_file = lbl_file.replace(".txt", ".jpg")
            img_path = os.path.join(cls_path, img_file)
            if not os.path.exists(img_path):
                img_file = lbl_file.replace(".txt", ".png")
                img_path = os.path.join(cls_path, img_file)
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            h, w = img.shape[:2]

            # Normalize
            x_center /= w
            y_center /= h
            w_box /= w
            h_box /= h

            class_id = class_to_id[cls]

            # Save YOLO label
            out_lbl_path = os.path.join(output_root, "labels", out_split, lbl_file)
            with open(out_lbl_path, "w") as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_box:.6f} {h_box:.6f}\n")

# Generate YAML
yaml_path = os.path.join(output_root, "farm_animals.yaml")
with open(yaml_path, "w") as f:
    f.write(f"path: ./{output_root}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("test: images/test\n\n")
    f.write("names:\n")
    for i, name in enumerate(class_names):
        f.write(f"  {i}: {name}\n")

print("✅ Conversion complete! YOLO dataset saved in", output_root)
