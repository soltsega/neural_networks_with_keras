# MNIST Handwritten Digit Classification

<div align="center">
  <img src="https://img.shields.io/badge/Status-Complete-success?style=for-the-badge" alt="Status: Complete">
  <img src="https://img.shields.io/badge/Accuracy-99.17%25-blue?style=for-the-badge" alt="Accuracy: 99.17%">
  <img src="https://img.shields.io/badge/Framework-TensorFlow-orange?style=for-the-badge" alt="Framework: TensorFlow">
</div>

## Project Overview
This repository contains a complete pipeline for classifying handwritten digits from the MNIST dataset using a Convolutional Neural Network (CNN). The project is structured into modular notebooks covering preprocessing, model design, training, and comprehensive evaluation.

## Performance Summary
- **Test Accuracy**: 99.17%
- **Test Loss**: 0.0254
- **Optimized for**: Precision and Recall across all digit classes (0-9).

## Repository Structure
- `data/`: Contains the preprocessed dataset in `.npz` format.
- `models/`: Stores the trained CNN model (`mnist_cnn_v1.h5`).
- `notebooks/`:
    - `01_preprocessing.ipynb`: Data loading, normalization, and reshaping.
    - `02_model_design_and_training.ipynb`: CNN architecture definition and training loop.
    - `03_evaluation_and_tuning.ipynb`: Performance metrics, error analysis, and hyperparameter experiments.
- `results/`: Contains performance charts, confusion matrices, and error visualizations.

## Model Architecture
The architecture is a sequential CNN designed for high spatial feature extraction:

<div align="center">
  <svg width="600" height="150" viewBox="0 0 600 150">
    <rect x="10" y="40" width="80" height="70" rx="5" fill="#e1f5fe" stroke="#01579b" stroke-width="2"/>
    <text x="50" y="80" text-anchor="middle" font-family="Arial" font-size="12" fill="#01579b">Input (28x28)</text>
    
    <line x1="90" y1="75" x2="120" y2="75" stroke="#01579bit" stroke-width="2" marker-end="url(#arrow)"/>
    
    <rect x="120" y="20" width="100" height="110" rx="5" fill="#f3e5f5" stroke="#4a148c" stroke-width="2"/>
    <text x="170" y="65" text-anchor="middle" font-family="Arial" font-size="11" fill="#4a148c">Conv2D (32)</text>
    <text x="170" y="85" text-anchor="middle" font-family="Arial" font-size="11" fill="#4a148c">ReLU + MaxPool</text>

    <line x1="220" y1="75" x2="250" y2="75" stroke="#01579b" stroke-width="2" />
    
    <rect x="250" y="20" width="100" height="110" rx="5" fill="#f3e5f5" stroke="#4a148c" stroke-width="2"/>
    <text x="300" y="65" text-anchor="middle" font-family="Arial" font-size="11" fill="#4a148c">Conv2D (64)</text>
    <text x="300" y="85" text-anchor="middle" font-family="Arial" font-size="11" fill="#4a148c">ReLU + MaxPool</text>

    <line x1="350" y1="75" x2="380" y2="75" stroke="#01579b" stroke-width="2" />
    
    <rect x="380" y="40" width="80" height="70" rx="5" fill="#fff3e0" stroke="#e65100" stroke-width="2"/>
    <text x="420" y="70" text-anchor="middle" font-family="Arial" font-size="11" fill="#e65100">Flatten +</text>
    <text x="420" y="90" text-anchor="middle" font-family="Arial" font-size="11" fill="#e65100">Dropout (0.5)</text>

    <line x1="460" y1="75" x2="490" y2="75" stroke="#01579b" stroke-width="2" />
    
    <rect x="490" y="50" width="100" height="50" rx="5" fill="#e8f5e9" stroke="#1b5e20" stroke-width="2"/>
    <text x="540" y="80" text-anchor="middle" font-family="Arial" font-size="11" fill="#1b5e20">Softmax Output</text>
    
    <defs>
      <marker id="arrow" markerWidth="10" markerHeight="10" refX="0" refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L9,3 z" fill="#01579b" />
      </marker>
    </defs>
  </svg>
</div>

## Preprocessing Details
- **Normalization**: Pixel values scaled to [0, 1] range.
- **Reshaping**: Images converted to (28, 28, 1) for grayscale convolutional input.
- **Encoding**: Labels converted to categorical one-hot vectors.

## Installation and Usage
1. Ensure the `.venv` environment is activated.
2. Install dependencies: `pip install -r requirements.txt`.
3. Follow the notebook sequence for a complete walkthrough:
    - `01_preprocessing.ipynb`: Prepare data.
    - `02_model_design_and_training.ipynb`: Train the model.
    - `03_evaluation_and_tuning.ipynb`: Analyze and tune results.
