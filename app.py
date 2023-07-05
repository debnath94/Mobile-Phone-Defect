# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 14:17:59 2023

@author: debna
"""

import streamlit as st
import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("model_mobilenetv2.h1")

labels = ['Good', 'ground_truth_1', 'ground_truth_2', 'Oil', 'Scratch','Stain']

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

def main():
    st.title("Phone Condition Detection")

    # Initialize the camera capture
    camera = cv2.VideoCapture(0)

    # Create a placeholder for displaying the camera frame
    frame_placeholder = st.empty()

    while True:
        # Capture frame from the camera
        ret, frame = camera.read()

        # Preprocess the frame
        processed_frame = preprocess_frame(frame)

        # Make predictions
        predictions = model.predict(processed_frame)
        predicted_label = labels[np.argmax(predictions)]

        # Display the predicted label on the frame
        cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Convert the frame to RGB for displaying in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in the Streamlit app
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera
    camera.release()

if __name__ == '__main__':
    main()












