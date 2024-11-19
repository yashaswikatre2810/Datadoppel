import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the CNN model
try:
    model = load_model('my_model.h5')
except Exception as e:
    st.error("Error loading model. Please ensure that 'my_model.h5' is in the correct location and is compatible.")
    st.stop()

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0
    return img

# Streamlit page configuration
st.set_page_config(page_title="Data Doppel", page_icon="🌀", layout="centered")

# Sidebar Navigation
st.sidebar.title("Data Doppel")
page = st.sidebar.selectbox("Navigation", ["Home", "Dataset", "Contact"])

# Home Page
if page == "Home":
    st.image("logo.png", width=100)
    st.title("Data Doppel")
    st.write("""
        Welcome to Data Doppel! This web application is designed to classify images using a Convolutional Neural Network (CNN) model.
        **Tech Stack:** Streamlit, TensorFlow, Python
        **Technology:** CNN (Convolutional Neural Network)
    """)

# Dataset Page
elif page == "Dataset":
    st.header("Upload and Classify an Image")

    # Step 1: Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image immediately after upload
        processed_img = preprocess_image(img)

        # Step 2: Classify the image and display prediction
        if st.button("Check Prediction and Confidence"):
            predictions = model.predict(processed_img)
            class_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100

            st.write(f"Predicted Class: {class_index}")
            st.write(f"Confidence: {confidence:.2f}%")

        # Step 3: Specify number of augmented images
        num_images = st.number_input("How many augmented images to generate?", min_value=1, max_value=10, step=1)

        # Step 4: Generate and display augmented images
        if st.button("Generate Augmented Images"):
            st.subheader("Generated Augmented Images")

            # Data augmentation to generate similar images
            data_gen = ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )

            fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
            img_iterator = data_gen.flow(processed_img, batch_size=1)

            for i in range(num_images):
                augmented_img = next(img_iterator)[0]
                axes[i].imshow(augmented_img)
                axes[i].axis('off')

            st.pyplot(fig)

# Contact Page
elif page == "Contact":
    st.header("Contact Us")
    st.write("**Address:** GHRCE College, Nagpur")
    st.write("**Mobile Number:** 7263049920")
    st.write("**Team Members:** Prashant, Shreerang, Mahek, Yashaswi")
