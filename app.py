import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import os

# Page Configuration
st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")

# --- CUSTOM STYLING (Emoji-free, Professional) ---
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #01579b;
        color: white;
    }
    .prediction-text {
        font-size: 80px;
        font-weight: bold;
        color: #01579b;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_mnist_model():
    model_path = 'models/mnist_cnn_v1.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_mnist_model()

# --- HEADER ---
st.title("MNIST Digit Classification")
st.write("An interactive demonstration of a Convolutional Neural Network (CNN) trained on the MNIST dataset.")

if model is None:
    st.error("Model file not found. Please ensure 'models/mnist_cnn_v1.h5' exists.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Technical Overview")
    st.info("System Status: Operational")
    st.metric(label="Validation Accuracy", value="99.17%")
    st.write("Architecture: CNN (2 Conv2D Layers)")
    st.write("Framework: TensorFlow/Keras")

# --- MAIN INTERFACE (TABS) ---
tab1, tab2 = st.tabs(["Upload Image", "Draw Digit"])

def preprocess_image(img):
    """Convert PIL image to model-ready numpy array."""
    # Convert to grayscale
    img = ImageOps.grayscale(img)
    # Resize to 28x28
    img = img.resize((28, 28))
    # Normalize to [0, 1]
    img_array = np.array(img).astype('float32') / 255.0
    # Reshape for CNN input (1, 28, 28, 1)
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

with tab1:
    st.subheader("Upload specialized digit image")
    uploaded_file = st.file_uploader("Choose a file (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=150)
        
        if st.button("Classify Uploaded Image"):
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)
            digit = np.argmax(prediction)
            confidence = np.max(prediction)
            
            st.markdown(f'<p class="prediction-text">{digit}</p>', unsafe_allow_html=True)
            st.write(f"Confidence: {confidence:.2%}")
            st.bar_chart(prediction[0])

with tab2:
    st.subheader("Draw a digit (0-9)")
    st.write("Use your mouse to draw a number in the box below.")
    
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        if st.button("Classify Drawing"):
            # canvas_result.image_data is a numpy array (RGBA)
            # We need to convert it to a PIL image first
            img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
            processed_img = preprocess_image(img)
            
            prediction = model.predict(processed_img)
            digit = np.argmax(prediction)
            confidence = np.max(prediction)
            
            st.markdown(f'<p class="prediction-text">{digit}</p>', unsafe_allow_html=True)
            st.write(f"Confidence: {confidence:.2%}")
            st.bar_chart(prediction[0])

# --- FOOTER ---
st.divider()
st.caption("Powered by TensorFlow and Streamlit. Documentation available in README.md.")
