import os
import random
import shutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import MobileNetV2
from PIL import Image  # ✅ use Pillow instead of imghdr

# ==========================
# STEP 1: Split Dataset into 70/15/15
# ==========================
original_dir = r"C:\Users\Dell\Downloads\animal ds seperate"   # dataset path
output_dir = "dataset_split"   # will be created in working dir

# Utility: check if file is a valid image
def is_image(file_path):
    try:
        Image.open(file_path).verify()  # will raise if not image
        return True
    except Exception:
        return False

# Create output folders
for split in ["train", "val", "test"]:
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    # Create class subfolders
    for class_name in os.listdir(original_dir):
        class_dir = os.path.join(split_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

# Split files
for class_name in os.listdir(original_dir):
    class_dir = os.path.join(original_dir, class_name)

    # Skip if not a folder (avoid files like "Label")
    if not os.path.isdir(class_dir):
        continue

    # ✅ Only include valid image files
    files = [
        f for f in os.listdir(class_dir)
        if os.path.isfile(os.path.join(class_dir, f)) and is_image(os.path.join(class_dir, f))
    ]

    random.shuffle(files)

    train_split = int(0.7 * len(files))
    val_split = int(0.85 * len(files))

    for i, file in enumerate(files):
        src = os.path.join(class_dir, file)
        if i < train_split:
            dst = os.path.join(output_dir, "train", class_name, file)
        elif i < val_split:
            dst = os.path.join(output_dir, "val", class_name, file)
        else:
            dst = os.path.join(output_dir, "test", class_name, file)

        shutil.copy(src, dst)

print("✅ Dataset successfully split into train/val/test (only valid image files copied)!")

# ==========================
# STEP 2: Load Datasets
# ==========================
img_size = (128, 128)
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(output_dir, "train"),
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical"
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(output_dir, "val"),
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical"
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(output_dir, "test"),
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical"
)

# 🔹 Capture class names BEFORE prefetch
class_names = train_ds.class_names
num_classes = len(class_names)

# Performance optimizations
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds  = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================
# EXPERIMENT TOGGLES
# ==========================
USE_AUG = True            # 🔹 Toggle Data Augmentation
USE_WEIGHT_DECAY = True   # 🔹 Toggle L2 Regularization (weight decay)
DROPOUT_RATE_1 = 0.3      # 🔹 First dropout (after GAP)
DROPOUT_RATE_2 = 0.2      # 🔹 Second dropout (after Dense)
BASE_LR = 1e-4            # 🔹 Learning rate (Phase 1)
FINE_TUNE_LR = 1e-5       # 🔹 Learning rate (Phase 2)
EPOCHS = 100              # 🔹 Number of epochs per phase

# ==========================
# STEP 3: Data Augmentation
# ==========================
if USE_AUG:
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
else:
    data_augmentation = keras.Sequential()  # identity layer

# ==========================
# STEP 4: Transfer Learning Model (MobileNetV2)
# ==========================
base_model = MobileNetV2(
    input_shape=img_size + (3,), 
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

# Apply weight decay if enabled
reg = regularizers.l2(1e-5) if USE_WEIGHT_DECAY else None

model = keras.Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(DROPOUT_RATE_1),
    layers.Dense(128, activation="relu", kernel_regularizer=reg),
    layers.Dropout(DROPOUT_RATE_2),
    layers.Dense(num_classes, activation="softmax")
])

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# ==========================
# Compile and Train
# ==========================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=BASE_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

# ==========================
# Plot Training Curves
# ==========================
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()

plt.show()

# ==========================
# Final Evaluation
# ==========================
test_loss, test_acc = model.evaluate(test_ds)
print(f"🎯 Final Test Accuracy: {test_acc:.4f}")

# Predictions
y_pred = model.predict(test_ds)
y_pred_classes = np.argmax(y_pred, axis=1)

# True labels
y_true = np.concatenate([np.argmax(y, axis=1) for x, y in test_ds], axis=0)

# Classification Report
print("📊 Classification Report:\n")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
