import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# 1. Load your trained model
model = tf.keras.models.load_model("model.h5")

# 2. Define class label mapping (same order as training)
class_names = {0: 'dry', 1: 'recyclable', 2: 'wet'}

# 3. Image preprocessing function
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 128, 128, 3)
    return img_array

# 4. Prediction function
def predict(img_path):
    img = prepare_image(img_path)
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    class_label = class_names[class_index]
    confidence = predictions[0][class_index]
    print(f"🧪 Prediction: {class_label} waste 🗑️  with {confidence*100:.2f}% confidence")

# 5. Run prediction on a sample image
predict("../dataset/dry/cardboard-dry.jpg")
