import streamlit as st
import numpy as np
from PIL import Image
import requests
import tensorflow as tf
import os

# --- 1. Download model from Google Drive link ---
MODEL_URL = "https://drive.google.com/uc?id=1jprgwZjV0_TTmXt-WsquAw-pbw1etRXZ"

MODEL_PATH = "best_model.keras"

@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.write("⏳ Downloading model from Google Drive...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        st.success("✅ Model downloaded.")
    else:
        st.success("✅ Model already exists.")
    return tf.keras.models.load_model(MODEL_PATH)

# --- 2. Load model ---
model = download_model()

# --- 3. Class names ---
class_names = ['Healthy (normal skin)', 'Unhealthy (ulcer detected)']

# --- 4. Preprocess image ---
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

# --- 5. Streamlit UI ---
st.set_page_config(page_title="Diabetic Foot Ulcer Detector", layout="centered")
st.title("🦶 Diabetic Foot Ulcer Detection")
st.write("Upload a foot image to detect presence of an ulcer.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write("📊 Analyzing...")
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f"🧠 **Prediction:** `{predicted_class}`")
