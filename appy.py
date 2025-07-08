import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model('model.h5')
input_shape = model.input_shape  # e.g. (None, 120, 120, 3) or (None, 57600)
class_names = {0: 'dry', 1: 'recyclable', 2: 'wet'}

def preprocess_image(img, model_input_shape):
    if len(model_input_shape) == 4:
        _, width, height, channels = model_input_shape
        img = img.resize((width, height))
        img_array = image.img_to_array(img) / 255.0
        
        if channels == 1 and img_array.shape[2] == 3:
            img_array = np.mean(img_array, axis=2, keepdims=True)
        elif channels == 3 and img_array.shape[2] == 1:
            img_array = np.repeat(img_array, 3, axis=2)
        
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    elif len(model_input_shape) == 2:
        input_dim = model_input_shape[1]
        side = int(round((input_dim / 3) ** 0.5))
        img = img.resize((side, side))
        img_array = image.img_to_array(img) / 255.0
        img_array = img_array.flatten()
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    else:
        raise ValueError(f"Unexpected model input shape: {model_input_shape}")

st.title("Smart Waste Segregation App")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Your uploaded image', use_container_width=True)
    
    try:
        processed_img = preprocess_image(img, input_shape)
        prediction = model.predict(processed_img)
        pred_class = np.argmax(prediction)
        confidence = prediction[0][pred_class] * 100

        st.write(f"Prediction: **{class_names[pred_class]}** waste")
        st.write(f"Confidence: **{confidence:.2f}%**")
    except Exception as e:
        st.error(f"Prediction error: {e}")
