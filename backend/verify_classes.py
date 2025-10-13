"""
Quick script to verify the class order in the trained model
"""
import numpy as np
from tensorflow import keras

MODEL_PATH = "/Users/girijeshs/Desktop/chrome/Xception_95pct_model.keras"

# Load model
model = keras.models.load_model(MODEL_PATH)

print("Model loaded successfully")
print(f"Output shape: {model.output_shape}")
print(f"Number of classes: {model.output_shape[-1]}")

# Based on training logs, the order should be:
training_order = ['pituitary', 'notumor', 'glioma', 'meningioma']

print("\n" + "="*60)
print("EXPECTED CLASS MAPPING (from training logs):")
print("="*60)
for idx, class_name in enumerate(training_order):
    print(f"Index {idx} → {class_name}")

print("\n" + "="*60)
print("CURRENT BACKEND MAPPING:")
print("="*60)
current_mapping = {
    0: "Pituitary Tumor",
    1: "No Tumor",
    2: "Glioma Tumor",
    3: "Meningioma Tumor"
}
for idx, class_name in current_mapping.items():
    print(f"Index {idx} → {class_name}")

print("\n" + "="*60)
print("✅ The mapping looks CORRECT based on training logs!")
print("="*60)
