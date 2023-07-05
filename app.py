# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 14:17:59 2023

@author: debna
"""


import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("model_mobilenetv2.h1")

# Define the labels (replace with your own labels)
labels = ['Good', 'ground_truth_1', 'ground_truth_2', 'Oil', 'Scratch', 'Stain']

# Set the maximum file size for image uploads (in bytes)
max_file_size = 10 * 1024 * 1024  # 10MB

def preprocess_image(image):
    # Resize the image to match the model input shape
    image = image.resize((100, 100))
    # Convert the image to a NumPy array
    image = np.array(image)
    # Scale the pixel values to the range of 0-1
    image = image / 255.0
    # Expand the dimensions to match the model input shape
    image = np.expand_dims(image, axis=0)
    return image

def predict(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    # Make predictions
    predictions = model.predict(processed_image)
    # Get the predicted label
    predicted_label = labels[np.argmax(predictions)]
    # Get the confidence score
    confidence = np.max(predictions)
    return predicted_label, confidence

# Set Streamlit app configurations
st.set_page_config(
    page_title="Mobile Phone Defect",
    layout="centered"
)

# Display the app title and description
st.title("Mobile Phone Defect Detection")
st.markdown("Upload an image of a mobile phone and the model will predict if it has any defects.")

# Create an uploader for image files
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

# Perform prediction if an image is uploaded
if uploaded_file is not None:
    # Ensure the uploaded file size is within the limit
    if uploaded_file.size <= max_file_size:
        # Load the image
        image = Image.open(uploaded_file)
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        # Perform prediction
        predicted_label, confidence = predict(image)
        # Display the predicted label and confidence score
        st.subheader("Prediction:")
        st.write(f"Defect: {predicted_label}")
        st.write(f"Confidence: {confidence:.2f}")
    else:
        st.error(f"The uploaded file exceeds the maximum file size of {max_file_size/(1024*1024)}MB.")












