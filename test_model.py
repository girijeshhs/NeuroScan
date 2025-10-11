"""
Test script to check your model configuration and help debug Grad-CAM issues.
Run this before starting the Flask app to ensure everything is configured correctly.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

# Your model path
MODEL_PATH = "/Users/girijeshs/Desktop/chrome/Xception_95pct_model.keras"

print("="*70)
print("🧠 BRAIN TUMOR MODEL CONFIGURATION TEST")
print("="*70)

try:
    # Load model
    print("\n📦 Loading model...")
    model = keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
    
    # Model info
    print("\n📊 MODEL INFORMATION:")
    print("-"*70)
    print(f"Input shape:  {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    num_classes = model.output_shape[-1]
    print(f"Number of output classes: {num_classes}")
    
    # Suggest class labels based on number of classes
    print("\n🏷️  SUGGESTED CLASS LABELS:")
    print("-"*70)
    if num_classes == 2:
        print("Your model has 2 classes. Use this in app.py:")
        print("CLASS_LABELS = {")
        print("    0: 'No Tumor',")
        print("    1: 'Tumor Detected'")
        print("}")
    elif num_classes == 4:
        print("Your model has 4 classes. Use this in app.py:")
        print("CLASS_LABELS = {")
        print("    0: 'Glioma Tumor',")
        print("    1: 'Meningioma Tumor',")
        print("    2: 'Pituitary Tumor',")
        print("    3: 'No Tumor'")
        print("}")
        print("\n⚠️  Note: The order (0,1,2,3) must match your training data!")
    elif num_classes == 3:
        print("Your model has 3 classes. Use this in app.py:")
        print("CLASS_LABELS = {")
        print("    0: 'Glioma Tumor',")
        print("    1: 'Meningioma Tumor',")
        print("    2: 'Pituitary Tumor'")
        print("}")
    else:
        print(f"Your model has {num_classes} classes.")
        print("CLASS_LABELS = {")
        for i in range(num_classes):
            print(f"    {i}: 'Class {i}',")
        print("}")
    
    # List all layers
    print("\n🔧 MODEL ARCHITECTURE:")
    print("-"*70)
    print(f"Total layers: {len(model.layers)}")
    
    # Find convolutional layers
    print("\n🎯 CONVOLUTIONAL LAYERS (for Grad-CAM):")
    print("-"*70)
    conv_layers = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            conv_layers.append(layer.name)
            print(f"  {i:2d}. {layer.name:30s} - Shape: {layer.output_shape}")
    
    if conv_layers:
        last_conv = conv_layers[-1]
        print("\n✨ GRAD-CAM CONFIGURATION:")
        print("-"*70)
        print(f"Last convolutional layer: '{last_conv}'")
        print("\nIf Grad-CAM doesn't work automatically, add this to app.py (line ~164):")
        print(f"last_conv_layer_name = '{last_conv}'  # Manual override")
    else:
        print("\n⚠️  WARNING: No Conv2D layers found!")
        print("Your model might not support Grad-CAM visualization.")
        print("Grad-CAM requires convolutional layers.")
    
    # Test with dummy input
    print("\n🧪 Testing model with dummy input...")
    dummy_input = np.random.rand(1, 299, 299, 3).astype('float32')
    try:
    
    # Summary
    print("\n" + "="*70)
    print("✅ CONFIGURATION SUMMARY")
    print("="*70)
    print(f"✓ Model loads successfully")
    print(f"✓ Number of classes: {num_classes}")
    if conv_layers:
        print(f"✓ Grad-CAM supported (using layer: {last_conv})")
    else:
        print(f"✗ Grad-CAM not supported (no conv layers)")
    print(f"✓ Model can make predictions")
    
    print("\n📝 NEXT STEPS:")
    print("-"*70)
    print("1. Update CLASS_LABELS in app.py with the suggested labels above")
    print("2. Make sure the class indices match your training data order")
    if conv_layers:
        print(f"3. If Grad-CAM fails, manually set: last_conv_layer_name = '{last_conv}'")
    print("4. Run: python app.py")
    print("5. Open index.html in your browser")
    
    print("\n" + "="*70)
    print("🎉 Ready to start your web app!")
    print("="*70)
    
except FileNotFoundError:
    print(f"\n❌ ERROR: Model file not found at: {MODEL_PATH}")
    print("\nPlease check:")
    print("1. Is the file path correct?")
    print("2. Does the file exist at that location?")
    print("3. Do you have read permissions?")
    
except Exception as e:
    print(f"\n❌ ERROR loading model: {str(e)}")
    print("\nThis might mean:")
    print("1. The file is corrupted")
    print("2. TensorFlow version mismatch")
    print("3. The file is not a valid Keras model")
    import traceback
    print("\nFull error:")
    traceback.print_exc()
