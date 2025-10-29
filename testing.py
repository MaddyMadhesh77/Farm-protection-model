import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

model = keras.models.load_model("saved_models/final_animal_classifier_2025-09-19_19-13-02.h5")

img_path = "dataset_split/test/Skunk/img_001.jpg"
img = load_img(img_path, target_size=img_size)
img_array = img_to_array(img) / 255.0  # scale as in training
img_array = np.expand_dims(img_array, axis=0)  # make batch of 1

pred = model.predict(img_array)
pred_class = np.argmax(pred, axis=1)[0]

print(f"Predicted class index: {pred_class}")
print(f"Predicted class name: {class_names[pred_class]}")

