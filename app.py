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
    
#classes for CIFAR-10 dataset
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
        response.raise_for_status()  # Ensure we got a valid response
        
        content = response.content
        st.write("Predicting Class...")
        with st.spinner("Classifying..."):
            img_tensor = load_image(content)
            pred = model.predict(img_tensor)
            pred_class = classes[np.argmax(pred)]
            st.write("Predicted Class:", pred_class)
            st.image(content, use_container_width=True)
    except requests.exceptions.RequestException as e:
        st.write("Failed to fetch image. Please check the URL.")
        st.write(f"Error: {e}")
    except Exception as e:
        st.write("An error occurred during processing.")
        st.write(f"Error: {e}")
