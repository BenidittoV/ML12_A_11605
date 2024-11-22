import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

model_path = r'best_model_tf.h5'

if os.path.exists(model_path):
    try:
        tf.get_logger().setLevel('ERROR')
        model = tf.keras.models.load_model(model_path, compile=False)

        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        def preprocess_image(image):
            image = image.resize((28, 28)) 
            image = image.convert('L') 
            image_array = np.array(image) / 255.0 
            image_array = image_array.reshape(1, 28, 28, 1) 
            return image_array

        st.title("Fashion MNIST Image Classifier 11605") 
        st.write("Unggah satu atau lebih gambar item fashion (misalnya sepatu, tas, baju), dan model akan memprediksi kelasnya.")

        uploaded_files = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        with st.sidebar:
            predict_button = st.button("Predict")

        if uploaded_files:
            images = [Image.open(file) for file in uploaded_files]
            st.image(images, caption=[file.name for file in uploaded_files], use_column_width=True)

        if predict_button:
            st.sidebar.write("### Hasil Prediksi")

            for file, image in zip(uploaded_files, images):
                processed_image = preprocess_image(image)
                predictions = model.predict(processed_image)[0]

                predicted_class = np.argmax(predictions)
                confidence = predictions[predicted_class] * 100 

                st.sidebar.write(f"### {file.name}")
                st.sidebar.write(f"Kelas Prediksi: **{class_names[predicted_class]}**")
                st.sidebar.write(f"Confidence: **{confidence:.2f}%**")
    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.error("File model tidak ditemukan.")
