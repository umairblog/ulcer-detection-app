
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model("best_model.keras")

# Class names
class_names = ['Healthy (normal skin)', 'Unhealthy (ulcer detected)']

# Image preprocessing
def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit UI
st.title("Diabetic Foot Ulcer Detection")
st.write("Upload an image to detect ulcer.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]

    st.write("### Prediction:")
    st.success(predicted_class)
