import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import base64
import io
import os
import time

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Neural Chat AI",
    page_icon="🤖",
    layout="wide"
)

# --- PREMIUM DASHBOARD CHAT CSS (Stationary) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* Global Aesthetics */
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }

    /* Conversation Bubble Styling */
    section[data-testid="stChatMessageContainer"] {
        padding: 1rem;
        background: transparent;
    }

    /* Custom Input Controls Panel */
    .input-panel {
        background: rgba(22, 27, 34, 0.95);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px;
        margin-top: 20px;
    }

    /* Sidebar Fix */
    section[data-testid="stSidebar"] {
        background-color: #010409;
        border-right: 1px solid #30363d;
    }

    /* Result Metric Highlight */
    .chat-prediction-card {
        padding: 16px;
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        margin-top: 10px;
    }
    
    .chat-digit {
        font-size: 2.5rem;
        font-weight: 800;
        color: #58a6ff;
    }

    /* Stationary Fix (Disable any extra Streamlit animations) */
    * { transition: none !important; animation: none !important; }

</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_engine():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'mnist_cnn_v1.h5')
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_engine()

# --- PREPROCESSING ---
def process_message_image(image):
    """Normalized image for AI vision."""
    img = ImageOps.grayscale(image)
    img = img.resize((28, 28), Image.LANCZOS)
    img_array = np.array(img).astype('float32') / 255.0
    if img_array.mean() > 0.5:
        img_array = 1.0 - img_array
    img_input = img_array.reshape(1, 28, 28, 1)
    return img_input, img_array

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am **Neural Vision AI**. Please provide a handwritten digit (0-9) via the tools in the sidebar, and I will classify it for you."}
    ]

# --- SIDEBAR TOOLS ---
with st.sidebar:
    st.markdown("### 🛠 Neural Tools")
    st.info("Use these tools to send a sample to the chat.")
    
    input_mode = st.radio("Select Input Mode", ["Upload Sample", "Handwritten Draw"])
    
    selected_image = None
    
    if input_mode == "Upload Sample":
        file = st.file_uploader("Drop image here", type=["png", "jpg", "jpeg"])
        if file:
            selected_image = Image.open(file)
            st.image(selected_image, caption="Current Target", width=150)
            
    else:
        # Drawing Canvas HTML
        CANVAS_HTML = """
        <div style="background:#0d1117; border:1px solid #30363d; border-radius:8px; padding:10px; text-align:center;">
            <canvas id="c" width="200" height="200" style="border:1px solid #58a6ff; background:black; cursor:crosshair;"></canvas>
            <br>
            <button onclick="D()" style="margin-top:10px; background:#238636; color:white; border:none; padding:8px 16px; border-radius:4px; cursor:pointer;">Capture</button>
            <button onclick="C()" style="background:#21262d; color:#c9d1d9; border:1px solid #30363d; padding:8px 16px; border-radius:4px; cursor:pointer;">Clear</button>
        </div>
        <script>
        const v = document.getElementById('c'), x = v.getContext('2d');
        let d = false;
        function C() { x.fillStyle='black'; x.fillRect(0,0,200,200); }
        C(); x.strokeStyle='white'; x.lineWidth=12; x.lineCap='round';
        function G(e) { const r=v.getBoundingClientRect(); return {x:(e.touches?e.touches[0].clientX:e.clientX)-r.left, y:(e.touches?e.touches[0].clientY:e.clientY)-r.top}; }
        v.onmousedown = v.ontouchstart = e => { e.preventDefault(); d=true; const p=G(e); x.beginPath(); x.moveTo(p.x,p.y); };
        v.onmousemove = v.ontouchmove = e => { e.preventDefault(); if(!d) return; const p=G(e); x.lineTo(p.x,p.y); x.stroke(); };
        v.onmouseup = v.ontouchend = v.onmouseleave = () => d=false;
        function D() { const l=document.createElement('a'); l.download='draw.png'; l.href=v.toDataURL(); l.click(); }
        </script>
        """
        components.html(CANVAS_HTML, height=320)
        st.caption("1. Draw above | 2. Click Capture | 3. Upload 'draw.png' in Upload Mode.")
        st.warning("Manual 'Capture' required due to canvas sandboxing.")

    if st.button("🚀 Analyze current Target") and selected_image:
        # Preprocess
        inp, vis = process_message_image(selected_image)
        # Classify
        probs = model.predict(inp, verbose=0)[0]
        digit = np.argmax(probs)
        conf = np.max(probs)
        
        # Append User Message (the image)
        st.session_state.messages.append({
            "role": "user",
            "image": selected_image,
            "content": "Analyze this digit for me."
        })
        
        # Append Assistant Response
        st.session_state.messages.append({
            "role": "assistant",
            "prediction": int(digit),
            "confidence": float(conf),
            "vision": vis,
            "content": f"I've analyzed the sample. I am **{conf:.1%}** confident that this is a **{digit}**."
        })
        st.rerun()

# --- MAIN CONVERSATION ---
st.markdown("## 🧠 Neural Chat Dashboard")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else None):
        # Text content
        st.markdown(msg["content"])
        
        # If user message has an image
        if "image" in msg:
            st.image(msg["image"], width=150)
            
        # If assistant message has technical analysis
        if "prediction" in msg:
            cols = st.columns([1, 2])
            with cols[0]:
                st.markdown(f"""
                <div class="chat-prediction-card">
                    <div style="font-size: 0.7rem; color: #8b949e; text-transform: uppercase;">Prediction</div>
                    <div class="chat-digit">{msg['prediction']}</div>
                </div>
                """, unsafe_allow_html=True)
            with cols[1]:
                st.markdown("##### AI Vision")
                st.image(msg["vision"], width=100, clamp=True)
                st.caption("How the Neural Network 'sees' the sample.")

# --- FOOTER ---
st.markdown("---")
st.caption("Conversational MNIST Classifier | Verified Stationary Design")
