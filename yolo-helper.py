# ==========================
# STEP 0: Install Packages
# ==========================
# pip install ultralytics scikit-learn seaborn matplotlib pandas

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO

# ==========================
# STEP 1: Dataset YAML
# ==========================
yaml_path = r"E:\vs\ml-dataset\FarmHarmfulAnimalYOLO\farm_animals.yaml"

# ==========================
# STEP 2: Train Model
# ==========================
def train_model():
    model = YOLO("yolov10s.pt")
    results = model.train(
        data=yaml_path,
        epochs=100,
        batch=16,
        imgsz=640,
        patience=10,
        optimizer="Adam",
        device=0,       # "cpu" if no GPU
        workers=0       # Windows needs workers=0
    )
    return model, results


# ==========================
# STEP 3: Detection-style Metrics
# ==========================
def evaluate_detection(model):
    metrics = model.val(split="test")  # test set evaluation
    print("✅ YOLO Detection Metrics (Overall):")
    print(metrics)   # overall precision, recall, mAP

    per_class = metrics.class_results
    print("\n📊 Per-Class Metrics:")
    print(f"{'Class':<15}{'Precision':<12}{'Recall':<12}{'mAP@50':<12}{'mAP@50-95':<12}")

    for cls_id, values in enumerate(per_class):
        cname = model.names[cls_id]
        precision, recall, map50, map5095 = values
        print(f"{cname:<15}{precision:<12.3f}{recall:<12.3f}{map50:<12.3f}{map5095:<12.3f}")


# ==========================
# STEP 4: Classification-style Evaluation (from .txt labels)
# ==========================
def evaluate_classification(model, test_images, test_labels):
    results = model.predict(source=test_images, save=False, conf=0.25)

    class_names = model.names
    y_true, y_pred = [], []

    for r in results:
        img_path = r.path
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_file = os.path.join(test_labels, base + ".txt")

        # ✅ Ground Truth (from YOLO .txt label file)
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                lines = f.readlines()
                if len(lines) > 0:
                    true_idx = int(lines[0].split()[0])  # take first class
                else:
                    continue
        else:
            continue
        y_true.append(true_idx)

        # ✅ Prediction (highest confidence box)
        if len(r.boxes) > 0:
            best_idx = int(r.boxes.conf.argmax())
            pred_idx = int(r.boxes.cls[best_idx].cpu().numpy())
        else:
            pred_idx = -1
        y_pred.append(pred_idx)

    # Remove invalid predictions
    valid_idx = [i for i, p in enumerate(y_pred) if p != -1]
    y_true = [y_true[i] for i in valid_idx]
    y_pred = [y_pred[i] for i in valid_idx]

    # ✅ Classification Report
    print("\n📊 Image-level Classification Report (from labels):\n")
    print(classification_report(y_true, y_pred, target_names=list(class_names.values())))

    # ✅ Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(class_names.values()),
                yticklabels=list(class_names.values()))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (YOLO - Image level, label-based)")
    plt.show()


# ==========================
# STEP 5: Training Curves
# ==========================
def plot_training_curves(model):
    log_path = os.path.join(model.trainer.save_dir, "results.csv")
    df = pd.read_csv(log_path)
    print("\n✅ Available columns in training log:")
    print(df.columns.tolist())

    plt.figure(figsize=(14, 6))

    # Loss Plot
    plt.subplot(1, 2, 1)
    if "train/box_loss" in df and "train/cls_loss" in df:
        plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss')
        plt.plot(df['epoch'], df['train/cls_loss'], label='Class Loss')
    if "val/box_loss" in df and "val/cls_loss" in df:
        plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
        plt.plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    # Metrics Plot
    plt.subplot(1, 2, 2)
    for metric in ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
        if metric in df:
            plt.plot(df['epoch'], df[metric], label=metric)
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title("Validation Metrics")
    plt.legend()

    plt.show()


# ==========================
# MAIN PIPELINE
# ==========================
def main():
    model, _ = train_model()
    evaluate_detection(model)

    # ✅ Pass both images and labels
    evaluate_classification(
        model,
        test_images=r"C:/Users/Dell/Downloads/animal_yolo/test/images",
        test_labels=r"C:/Users/Dell/Downloads/animal_yolo/test/labels"
    )

    plot_training_curves(model)


if __name__ == "__main__":
    main()
