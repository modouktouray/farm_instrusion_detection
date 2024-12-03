import streamlit as st
import tensorflow as tf
from tensorflow import keras
import requests
import numpy as np

#Title
st.title("Farm intrusion detection")

#load model, set cache to prevent reloading
@st.cache_resource
def load_model():
    model=tf.keras.models.load_model('models/inceptionV3_model.h5')
    return model

with st.spinner("Loading Model...."):
    model=load_model()
    
#classes
classes=["humans", "monkeys"]

# image preprocessing
def load_image(image):
    img=tf.image.decode_jpeg(image,channels=3)
    img=tf.cast(img,tf.float32)
    img/=255.0
    img=tf.image.resize(img,(299,299))
    img=tf.expand_dims(img,axis=0)
    return img

#Get image URL from user
image_path = st.text_input("Enter Image URL to classify...")
# Get image from URL and predict
if image_path:
    try:
        # Fetch the image content
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(image_path, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        content = response.content
        st.write("Predicting class...")
        
        with st.spinner("Classifying..."):
            # Preprocess and predict
            img_tensor = load_image(content)
            pred = model.predict(img_tensor)[0][0]  # Get prediction
            pred_class = classes[1] if pred > 0.5 else classes[0]  # Threshold at 0.5
            confidence = pred if pred > 0.5 else 1 - pred
            
            # Display results
            st.write(f"**Predicted Class:** {pred_class}")
            st.write(f"**Confidence:** {confidence * 100:.2f}%")
            st.image(content, caption=f"Classified as: {pred_class}", use_container_width=True)
    except requests.exceptions.RequestException as e:
        st.error("Failed to fetch image. Please check the URL.")
        st.error(f"Error: {e}")
    except Exception as e:
        st.error("An error occurred during processing.")
        st.error(f"Error: {e}")