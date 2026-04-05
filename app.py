import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import base64
import io
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="MNIST Neural Vision",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PREMIUM STATIONARY CSS (No Animations) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* Main Container Styling */
    .main {
        background-color: #0b0e14;
        color: #e6edf3;
        font-family: 'Inter', sans-serif;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(23, 27, 33, 0.7);
        border: 1px solid rgba(48, 54, 61, 1);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(8px);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #30363d;
    }

    /* Typography */
    h1, h2, h3 {
        color: #f0f6fc;
        font-weight: 700;
    }
    
    .hero-text {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #58a6ff, #bc8cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }

    /* Custom Metric Styling */
    .metric-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 15px;
        background: rgba(31, 35, 41, 1);
        border-radius: 8px;
        border: 1px solid #30363d;
    }
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        color: #58a6ff;
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        color: #8b949e;
        letter-spacing: 1px;
    }

    /* System Badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        margin-right: 8px;
    }
    .badge-online { background: rgba(35, 134, 54, 0.2); color: #3fb950; border: 1px solid #238636; }
    .badge-test { background: rgba(31, 111, 235, 0.2); color: #58a6ff; border: 1px solid #388bfd; }

    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: transparent;
        border: 1px solid #30363d;
        border-radius: 8px;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f2329 !important;
        border-color: #58a6ff !important;
        color: #f0f6fc !important;
    }

    /* Prediction Preview Mask */
    .ai-vision-box {
        border: 4px solid #30363d;
        border-radius: 8px;
        image-rendering: pixelated;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(180deg, #21262d 0%, #161b22 100%);
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 6px;
        font-weight: 600;
        width: 100%;
        transition: none; /* No Animation */
    }
    .stButton>button:hover {
        border-color: #8b949e;
        color: #f0f6fc;
    }

</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_mnist_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'mnist_cnn_v1.h5')
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_mnist_model()

# --- HEADER SECTION ---
col_head, col_space = st.columns([3, 1])
with col_head:
    st.markdown('<p class="hero-text">Neural Vision</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom: 30px;">
        <span class="status-badge badge-online">System: Operational</span>
        <span class="status-badge badge-test">Accuracy: 99.17%</span>
    </div>
    """, unsafe_allow_html=True)

# --- SIDEBAR DASHBOARD ---
with st.sidebar:
    st.markdown("### System Dashboard")
    st.markdown("---")
    st.markdown("**Engine**: TensorFlow / Keras")
    st.markdown("**Model Version**: MNIST-CNN v1.0")
    st.markdown("**Layer Count**: 7 (2 Convolutional)")
    st.markdown("---")
    st.markdown("### Instructions")
    st.caption("1. Choose input method (Upload/Draw)")
    st.caption("2. Review the 'AI Vision' preview")
    st.caption("3. Click 'Classify' to see prediction")
    st.markdown("---")
    st.markdown("Built with **DeepMind AntiGravity** concepts.")

if model is None:
    st.error("Model engine is offline. Please ensure models/mnist_cnn_v1.h5 exists.")
    st.stop()

# --- PREPROCESSING UTILITY ---
def get_ai_vision(image):
    """Processes image and returns both the 4D model input AND the 28x28 grayscale preview."""
    img = ImageOps.grayscale(image)
    # MNIST images are white digits on black. 
    # If the user uploaded a black digit on white, we might need to invert. 
    # For now, we standardize and resize.
    img = img.resize((28, 28), Image.LANCZOS)
    img_array = np.array(img).astype('float32') / 255.0
    
    # Simple check: if mean is > 0.5, it's likely a white background, so invert for MNIST
    if img_array.mean() > 0.5:
        img_array = 1.0 - img_array
        
    img_input = img_array.reshape(1, 28, 28, 1)
    return img_input, img_array

# --- MAIN INTERFACE ---
tab_upload, tab_draw = st.tabs(["[ 📁 ] Upload Image", "[ 🎨 ] Draw Digit"])

# -- CANVAS COMPONENT HTML --
# Note: Same robust HTML5 canvas as before, but with updated cleaner styling
CANVAS_HTML = """
<div style="display:flex;flex-direction:column;align-items:center;gap:15px; background:#0d1117; padding:20px; border-radius:12px; border:1px solid #30363d;">
    <canvas id="drawCanvas" width="280" height="280"
        style="border:2px solid #58a6ff; border-radius:8px; cursor:crosshair; touch-action:none; background:#000;">
    </canvas>
    <div style="display:flex;gap:15px;">
        <button onclick="clearCanvas()"
            style="padding:10px 30px; border:1px solid #30363d; border-radius:8px; font-size:14px; font-weight:600;
            cursor:pointer; background:#21262d; color:#c9d1d9;">Clear</button>
        <button onclick="submitDrawing()"
            style="padding:10px 30px; border:none; border-radius:8px; font-size:14px; font-weight:600;
            cursor:pointer; background:#238636; color:white;">Capture Drawing</button>
    </div>
</div>
<script>
const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
let drawing = false;

function clearCanvas() {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, 280, 280);
}
clearCanvas();

ctx.strokeStyle = '#FFF';
ctx.lineWidth = 18;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

function getPos(e) {
    const r = canvas.getBoundingClientRect();
    const x = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
    const y = (e.touches ? e.touches[0].clientY : e.clientY) - r.top;
    return {x, y};
}

canvas.addEventListener('mousedown', e => { e.preventDefault(); drawing=true; const p=getPos(e); ctx.beginPath(); ctx.moveTo(p.x,p.y); });
canvas.addEventListener('mousemove', e => { e.preventDefault(); if(!drawing)return; const p=getPos(e); ctx.lineTo(p.x,p.y); ctx.stroke(); });
canvas.addEventListener('mouseup', e => { e.preventDefault(); drawing=false; });
canvas.addEventListener('mouseleave', e => { drawing=false; });
canvas.addEventListener('touchstart', e => { e.preventDefault(); drawing=true; const p=getPos(e); ctx.beginPath(); ctx.moveTo(p.x,p.y); });
canvas.addEventListener('touchmove', e => { e.preventDefault(); if(!drawing)return; const p=getPos(e); ctx.lineTo(p.x,p.y); ctx.stroke(); });
canvas.addEventListener('touchend', e => { e.preventDefault(); drawing=false; });

function submitDrawing() {
    const dataUrl = canvas.toDataURL('image/png');
    const link = document.createElement('a');
    link.download = 'capture.png';
    link.href = dataUrl;
    link.click();
}
</script>
"""

def classification_ui(image):
    """Reusable UI for displaying results with AI Vision preview."""
    input_4d, preview_2d = get_ai_vision(image)
    
    col_vision, col_results = st.columns([1, 2])
    
    with col_vision:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### AI Vision")
        st.caption("How the neural network sees the input (28x28 normalized grayscale)")
        st.image(preview_2d, use_container_width=True, clamp=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_results:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Inference Result")
        
        # Perform prediction
        probs = model.predict(input_4d, verbose=0)[0]
        digit = np.argmax(probs)
        confidence = np.max(probs)
        
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Predicted Digit</div>
                <div class="metric-value">{digit}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with res_col2:
            st.markdown(f"""
            <div class="metric-container" style="height: 100%;">
                <div class="metric-label">Confidence</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #3fb950; margin-top: 10px;">{confidence:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        st.markdown("##### Probabilities")
        chart_data = {str(i): float(probs[i]) for i in range(10)}
        st.bar_chart(chart_data)
        st.markdown('</div>', unsafe_allow_html=True)

with tab_upload:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### Upload Target")
    uploaded_file = st.file_uploader("Select PNG, JPG, or JPEG for classification", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        if st.button("[ 🔋 ] Classify Uploaded Sample"):
            classification_ui(image)
    st.markdown('</div>', unsafe_allow_html=True)

with tab_draw:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### Neural Canvas")
    st.info("Draw a digit inside the blue square. Click 'Capture Drawing' to generate a sample, then upload it in the 'Upload Image' tab.")
    components.html(CANVAS_HTML, height=450)
    st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<div style="text-align:center; padding: 20px; color: #8b949e; font-size: 0.8rem;">
    MNIST Neural Vision | DeepMind AntiGravity Edition | (c) 2026
</div>
""", unsafe_allow_html=True)
