import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Ensure results directory exists
if not os.path.exists('results'):
    os.makedirs('results')

# Load data and model
data = np.load('data/mnist_preprocessed.npz')
x_test, y_test = data['x_test'], data['y_test']
model = tf.keras.models.load_model('models/mnist_cnn_v1.h5')

# Get predictions
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# 1. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - MNIST CNN')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Error Analysis (Misclassified Samples)
errors = np.where(y_pred != y_true)[0]
plt.figure(figsize=(12, 6))
for i, error_idx in enumerate(errors[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[error_idx].reshape(28, 28), cmap='gray')
    plt.title(f"True: {y_true[error_idx]}\nPred: {y_pred[error_idx]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig('results/error_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Results exported successfully to results/ folder.")
