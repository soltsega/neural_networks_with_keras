# Neural Vision Pro: High-Performance MNIST Classification

<div align="center">
  <img src="https://img.shields.io/badge/Status-In--Development-orange?style=for-the-badge" alt="Status: In Development">
  <img src="https://img.shields.io/badge/Accuracy-99.17%25-blue?style=for-the-badge" alt="Accuracy: 99.17%">
  <img src="https://img.shields.io/badge/UI-Neural--Vision--Pro-bc8cff?style=for-the-badge" alt="UI: Neural Vision Pro">
</div>

## 🧠 Project Overview
**Neural Vision Pro** is a comprehensive deep learning suite designed for high-precision handwritten digit classification. Built on the MNIST dataset, this project goes beyond simple training by providing a professional-grade **Inference Laboratory** for real-time model interaction, technical visualization, and performance auditing.

## 🚀 Key Features
- **CNN Architecture**: A multi-layered Convolutional Neural Network achieving a stable **99.17% test accuracy**.
- **Neural Vision Microscope**: Real-time visualization of how the AI "sees" your input (28x28 normalized grayscale).
- **Inference Laboratory**: A dual-input dashboard supporting both **Manual Drawing** and **Image Uploads**.
- **Premium Stationary UI**: A high-contrast, professional-grade interface built with Streamlit, optimized for web deployment.
- **Technical Metrics**: Real-time probability distribution mapping and confidence scoring for every prediction.

## 🏗 Model Architecture
The engine is a sequential CNN optimized for spatial feature extraction:

<div align="center">
  <svg width="600" height="150" viewBox="0 0 600 150">
    <rect x="10" y="40" width="80" height="70" rx="5" fill="#0d1117" stroke="#58a6ff" stroke-width="2"/>
    <text x="50" y="80" text-anchor="middle" font-family="Arial" font-size="12" fill="#58a6ff">Input (28x28)</text>
    <rect x="120" y="20" width="100" height="110" rx="5" fill="#161b22" stroke="#bc8cff" stroke-width="2"/>
    <text x="170" y="65" text-anchor="middle" font-family="Arial" font-size="11" fill="#bc8cff">Conv2D (32)</text>
    <text x="170" y="85" text-anchor="middle" font-family="Arial" font-size="11" fill="#bc8cff">ReLU + MaxPool</text>
    <rect x="250" y="20" width="100" height="110" rx="5" fill="#161b22" stroke="#bc8cff" stroke-width="2"/>
    <text x="300" y="65" text-anchor="middle" font-family="Arial" font-size="11" fill="#bc8cff">Conv2D (64)</text>
    <text x="300" y="85" text-anchor="middle" font-family="Arial" font-size="11" fill="#bc8cff">ReLU + MaxPool</text>
    <rect x="380" y="40" width="80" height="70" rx="5" fill="#21262d" stroke="#58a6ff" stroke-width="2"/>
    <text x="420" y="70" text-anchor="middle" font-family="Arial" font-size="11" fill="#58a6ff">Flatten +</text>
    <text x="420" y="90" text-anchor="middle" font-family="Arial" font-size="11" fill="#58a6ff">Dropout (0.5)</text>
    <rect x="490" y="50" width="100" height="50" rx="5" fill="#238636" stroke="#2ea043" stroke-width="2"/>
    <text x="540" y="80" text-anchor="middle" font-family="Arial" font-size="11" fill="#ffffff">Softmax Output</text>
  </svg>
</div>

## 📊 Performance Results
| Metric | Value |
| :--- | :--- |
| **Test Accuracy** | 99.17% |
| **Test Loss** | 0.0254 |
| **Inference Latency** | ~32ms |
| **Framework** | TensorFlow 2.x |

## 🛠 Installation & Local Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/soltsega/neural_networks.git
   cd neural_networks
   ```
2. **Environment Setup**:
   Ensure you have Python 3.11 installed. Use the provided `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the Dashboard**:
   ```bash
   streamlit run app.py
   ```

## 🌐 Cloud Deployment
Neural Vision Pro is optimized for **Streamlit Cloud**. 
- **Configuration**: The project is pinned to **Python 3.11** via `.python-version` to ensure maximum compatibility with the TensorFlow engine.
- **Current Status**: Active [Link Placeholder - Deployment Pending]

## 📂 Repository Structure
- `app.py`: The primary Neural Vision Pro dashboard.
- `notebooks/`: Modular pipeline from preprocessing to hyperparameter tuning.
- `models/`: Production-ready H5 weights.
- `canvas_component/`: Custom HTML5 canvas engine for zero-dependency drawing.

---
*Developed with a focus on Neural Precision and Professional UX.*
