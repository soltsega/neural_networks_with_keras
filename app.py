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
    page_title="Neural Vision Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- ADVANCED PREMIUM CSS (No Animations) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* Global Aesthetics */
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }

    /* High-Contrast Sharp Cards */
    .glass-card {
        background: #161b22;
        border: 1px solid #444c56;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 24px;
    }

    /* Hero Section */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #58a6ff, #bc8cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    
    .hero-subtitle {
        font-size: 1rem;
        color: #8b949e;
        margin-bottom: 30px;
    }

    /* Technical Stats Grid - High Contrast */
    .stat-card {
        padding: 15px;
        background: #21262d;
        border: 1px solid #444c56;
        border-radius: 6px;
        text-align: center;
    }
    .stat-label { font-size: 0.7rem; text-transform: uppercase; color: #8b949e; letter-spacing: 1px; }
    .stat-value { font-size: 1.2rem; font-weight: 700; color: #f0f6fc; }

    /* Prediction Metric - Sharp Focus */
    .prediction-box {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 30px;
        background: #0d1117;
        border: 2px solid #58a6ff;
        border-radius: 12px;
    }
    .prediction-digit {
        font-size: 6rem;
        font-weight: 900;
        color: #58a6ff;
        line-height: 1;
    }
    .prediction-label { color: #f0f6fc; font-weight: 600; margin-top: 10px; }

    /* AI Vision High-End */
    .vision-preview {
        border: 4px solid #30363d;
        border-radius: 8px;
        image-rendering: pixelated;
    }

    /* Custom Buttons (Stationary) */
    .stButton>button {
        background-color: #238636;
        color: white;
        border: 1px solid #2ea043;
        border-radius: 6px;
        font-weight: 700;
        height: 3.5rem;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        color: white;
    }

    /* Probability List - High Contrast */
    .prob-row {
        display: flex;
        align-items: center;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
    .prob-label { width: 30px; font-weight: 700; color: #8b949e; }
    .prob-bar-container { flex-grow: 1; background: #21262d; height: 10px; border-radius: 5px; margin: 0 12px; overflow: hidden; border: 1px solid #30363d; }
    .prob-bar-fill { height: 100%; background: #58a6ff; }
    .prob-value { width: 50px; text-align: right; color: #f0f6fc; font-weight: 700; }

</style>
""", unsafe_allow_html=True)

# --- ENGINE ---
@st.cache_resource
def load_engine():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'mnist_cnn_v1.h5')
    return tf.keras.models.load_model(p) if os.path.exists(p) else None

model = load_engine()

# --- PREPROCESSING ---
def get_ai_vision(image):
    img = ImageOps.grayscale(image)
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img).astype('float32') / 255.0
    if arr.mean() > 0.5: arr = 1.0 - arr
    return arr.reshape(1, 28, 28, 1), arr

# --- INTERFACE ---
st.markdown('<p class="hero-title">Neural Vision Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">High-Performance Convolutional Neural Network Analysis Interface</p>', unsafe_allow_html=True)

if model is None:
    st.error("Engine Offline: models/mnist_cnn_v1.h5 not found.")
    st.stop()

# -- TECH STATS ROW --
s1, s2, s3, s4 = st.columns(4)
with s1: st.markdown('<div class="stat-card"><div class="stat-label">System</div><div class="stat-value">Operational</div></div>', unsafe_allow_html=True)
with s2: st.markdown('<div class="stat-card"><div class="stat-label">Model</div><div class="stat-value">CNN v1.0</div></div>', unsafe_allow_html=True)
with s3: st.markdown('<div class="stat-card"><div class="stat-label">Accuracy</div><div class="stat-value">99.17%</div></div>', unsafe_allow_html=True)
with s4: st.markdown('<div class="stat-card"><div class="stat-label">Latency</div><div class="stat-value">32ms</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -- MAIN LAYOUT --
main_col, side_col = st.columns([1.8, 1])

with main_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### Input Laboratory")
    
    tab_draw, tab_upload = st.tabs(["[ 🎨 ] Handwritten Draw", "[ 📁 ] Upload Sample"])
    
    with tab_draw:
        # Improved Canvas UI
        CANVAS_HTML = """
        <div style="background:#010409; border:1px solid #30363d; border-radius:12px; padding:25px; text-align:center;">
            <canvas id="c" width="300" height="300" style="border:2px solid #58a6ff; border-radius:8px; background:black; cursor:crosshair;"></canvas>
            <div style="margin-top:20px; display:flex; gap:10px; justify-content:center;">
                <button onclick="D()" style="background:#238636; color:white; border:none; padding:12px 30px; border-radius:6px; font-weight:700; cursor:pointer;">Capture Image</button>
                <button onclick="C()" style="background:#21262d; color:#c9d1d9; border:1px solid #30363d; padding:12px 30px; border-radius:6px; font-weight:700; cursor:pointer;">Reset</button>
            </div>
        </div>
        <script>
        const v=document.getElementById('c'), x=v.getContext('2d'); let d=false;
        function C(){x.fillStyle='black'; x.fillRect(0,0,300,300);} C();
        x.strokeStyle='white'; x.lineWidth=20; x.lineCap='round';
        function G(e){const r=v.getBoundingClientRect(); return{x:(e.touches?e.touches[0].clientX:e.clientX)-r.left, y:(e.touches?e.touches[0].clientY:e.clientY)-r.top};}
        v.onmousedown=v.ontouchstart=e=>{e.preventDefault(); d=true; const p=G(e); x.beginPath(); x.moveTo(p.x,p.y);};
        v.onmousemove=v.ontouchmove=e=>{e.preventDefault(); if(!d)return; const p=G(e); x.lineTo(p.x,p.y); x.stroke();};
        v.onmouseup=v.ontouchend=v.onmouseleave=()=>d=false;
        function D(){const l=document.createElement('a'); l.download='capture.png'; l.href=v.toDataURL(); l.click();}
        </script>
        """
        components.html(CANVAS_HTML, height=450)
        st.info("Capture your signature digit and upload it in the next tab to verify.")
        
    with tab_upload:
        sample = st.file_uploader("Upload pre-captured digit", type=["png","jpg","jpeg"])
        if sample:
            img = Image.open(sample)
            st.image(img, caption="Loaded Sample", width=200)
            if st.button("🚀 EXECUTE INFERENCE"):
                st.session_state.current_sample = img
    st.markdown('</div>', unsafe_allow_html=True)

with side_col:
    if "current_sample" in st.session_state:
        # Perform Prediction
        inp, vis = get_ai_vision(st.session_state.current_sample)
        probs = model.predict(inp, verbose=0)[0]
        digit = np.argmax(probs)
        conf = np.max(probs)

        # Result Card
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🔮 Inference Result")
        st.markdown(f"""
        <div class="prediction-box">
            <div class="prediction-label">PREDICTED DIGIT</div>
            <div class="prediction-digit">{digit}</div>
            <div style="color:#3fb950; font-weight:700;">{conf:.2%} CONFIDENCE</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # AI Vision Card
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🔬 Neural Vision")
        st.caption("Normalized input (28x28 grayscale) injected into the CNN.")
        st.image(vis, use_container_width=True, clamp=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Probabilities Card
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Distribution")
        for i in range(10):
            p = float(probs[i])
            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-label">{i}</div>
                <div class="prob-bar-container"><div class="prob-bar-fill" style="width: {p*100}%; background: {'#3fb950' if i==digit else '#58a6ff'};"></div></div>
                <div class="prob-value">{p:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="glass-card" style="text-align:center; padding: 100px 20px;">', unsafe_allow_html=True)
        st.markdown("### ⌛ AWAITING INPUT")
        st.caption("Upload or Capture a sample to begin inference analysis.")
        st.markdown('</div>', unsafe_allow_html=True)

# -- FOOTER --
st.caption("Neural Vision Pro v1.0 | Stationary UX Engine | (c) 2026")
