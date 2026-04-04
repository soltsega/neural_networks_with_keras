import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="MNIST Digit Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e0e0;
    }

    .block-container {
        padding-top: 2rem;
    }

    .header-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .header-subtitle {
        font-size: 1rem;
        color: #8892b0;
        text-align: center;
        margin-bottom: 2rem;
    }

    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }

    .prediction-digit {
        font-size: 96px;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }

    .confidence-label {
        font-size: 1.1rem;
        color: #a8b2d1;
        margin-top: 0.5rem;
    }

    .confidence-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #64ffda;
    }

    .stat-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }

    .stat-label {
        font-size: 0.75rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stat-value {
        font-size: 1.3rem;
        font-weight: 600;
        color: #ccd6f6;
    }

    div[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        justify-content: center;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 24px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #a8b2d1;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: 600;
        font-size: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }

    footer {
        color: #4a5568;
    }
</style>
""", unsafe_allow_html=True)


# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'mnist_cnn_v1.h5')
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None


model = load_model()

# --- HEADER ---
st.markdown('<p class="header-title">MNIST Digit Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="header-subtitle">Real-time handwritten digit recognition powered by a Convolutional Neural Network</p>', unsafe_allow_html=True)

if model is None:
    st.error("Model file not found at models/mnist_cnn_v1.h5. Train the model first using the notebooks.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### System Dashboard")
    st.markdown("---")

    st.markdown("""
    <div class="stat-box">
        <div class="stat-label">Model Status</div>
        <div class="stat-value" style="color: #64ffda;">Loaded</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-box">
        <div class="stat-label">Test Accuracy</div>
        <div class="stat-value">99.17%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-box">
        <div class="stat-label">Architecture</div>
        <div class="stat-value">CNN (2 Conv Layers)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-box">
        <div class="stat-label">Framework</div>
        <div class="stat-value">TensorFlow / Keras</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="stat-box">
        <div class="stat-label">Input Shape</div>
        <div class="stat-value">28 x 28 x 1</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-box">
        <div class="stat-label">Output Classes</div>
        <div class="stat-value">10 (Digits 0-9)</div>
    </div>
    """, unsafe_allow_html=True)


# --- PREPROCESSING ---
def preprocess_image(img):
    """Convert any image to model-ready 28x28 grayscale numpy array."""
    img = ImageOps.grayscale(img)
    img = ImageOps.invert(img)  # MNIST expects white digit on black background
    img = img.resize((28, 28), Image.LANCZOS)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array


def preprocess_canvas(canvas_data):
    """Convert canvas RGBA numpy array to model-ready input."""
    # Extract alpha channel (the drawing strokes)
    img = Image.fromarray(canvas_data.astype('uint8'), 'RGBA')
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28), Image.LANCZOS)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array


def display_results(prediction):
    """Display prediction results with styled layout."""
    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    col_result, col_chart = st.columns([1, 2])

    with col_result:
        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-digit">{digit}</div>
            <div class="confidence-label">Confidence</div>
            <div class="confidence-value">{confidence:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_chart:
        st.markdown("#### Probability Distribution")
        chart_data = {str(i): float(prediction[0][i]) for i in range(10)}
        st.bar_chart(chart_data)


# --- TABS ---
tab_upload, tab_draw = st.tabs(["Upload Image", "Draw Digit"])

with tab_upload:
    st.markdown("#### Upload a digit image for classification")

    uploaded_file = st.file_uploader(
        "Supported formats: PNG, JPG, JPEG",
        type=["png", "jpg", "jpeg"],
        key="uploader"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col_img, col_space = st.columns([1, 3])
        with col_img:
            st.image(image, caption="Uploaded Image", width=180)

        if st.button("Classify", key="btn_upload"):
            with st.spinner("Analyzing..."):
                processed = preprocess_image(image)
                prediction = model.predict(processed, verbose=0)
                display_results(prediction)

with tab_draw:
    st.markdown("#### Draw a single digit (0-9) in the canvas below")
    st.markdown("Use your mouse or touchscreen to draw. Click **Classify Drawing** when ready.")

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=300,
        width=300,
        drawing_mode="freedraw",
        update_streamlit=True,
        key="draw_canvas",
    )

    if st.button("Classify Drawing", key="btn_draw"):
        if canvas_result is not None and canvas_result.image_data is not None:
            canvas_array = canvas_result.image_data
            has_drawing = np.any(canvas_array[:, :, :3] > 0)

            if has_drawing:
                with st.spinner("Analyzing..."):
                    processed = preprocess_canvas(canvas_array)
                    prediction = model.predict(processed, verbose=0)
                    display_results(prediction)
            else:
                st.warning("The canvas is empty. Please draw a digit first.")
        else:
            st.warning("Canvas has not loaded yet. Please try again.")

# --- FOOTER ---
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#4a5568; font-size:0.85rem;">'
    'MNIST CNN Classifier -- Built with TensorFlow and Streamlit'
    '</p>',
    unsafe_allow_html=True
)
