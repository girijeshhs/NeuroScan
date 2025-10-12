# Model Configuration Guide

## 🔧 IMPORTANT: Configure Your Tumor Classes

Your model's output determines what tumor types can be detected. You need to update `app.py` to match your model's training.

## How to Find Your Model's Classes

Run this Python script to check your model:

```python
import tensorflow as tf

# Load your model
model = tf.keras.models.load_model("/Users/girijeshs/Downloads/Brave/VGG16_final.keras")

# Check output shape
print(f"Model output shape: {model.output_shape}")
print(f"Number of classes: {model.output_shape[-1]}")

# If you have access to your training code, check the class names used during training
```

## Common Configurations

### Option 1: Binary Classification (2 classes)
If your model outputs 2 classes:

```python
CLASS_LABELS = {
    0: "No Tumor",
    1: "Tumor Detected"
}
```

### Option 2: 4-Class Brain Tumor Classification (Most Common)
If your model outputs 4 classes:

```python
CLASS_LABELS = {
    0: "Glioma Tumor",
    1: "Meningioma Tumor",
    2: "Pituitary Tumor",
    3: "No Tumor"
}
```

### Option 3: 3-Class Tumor Types Only
If your model outputs 3 tumor types:

```python
CLASS_LABELS = {
    0: "Glioma Tumor",
    1: "Meningioma Tumor",
    2: "Pituitary Tumor"
}
```

## How to Update app.py

1. Open `app.py`
2. Find this section (around line 16-25):

```python
CLASS_LABELS = {
    0: "Glioma Tumor",
    1: "Meningioma Tumor", 
    2: "Pituitary Tumor",
    3: "No Tumor"
}
```

3. Replace with the correct mapping based on your model's training
4. **IMPORTANT**: The numbers (0, 1, 2, 3) must match the order your model was trained with!

## About Brain Tumor Types

### Glioma
- Most common type of brain tumor
- Originates from glial cells
- Can be low-grade or high-grade (aggressive)
- Color: Red

### Meningioma
- Usually benign (non-cancerous)
- Grows from meninges (brain/spinal cord membranes)
- Most common benign brain tumor
- Color: Orange

### Pituitary Tumor
- Grows in the pituitary gland
- Can affect hormone production
- Usually benign
- Color: Purple

## Troubleshooting Grad-CAM Issues

If Grad-CAM is not showing:

1. **Check model layers**: Run this to see your model's convolutional layers:

```python
import tensorflow as tf

model = tf.keras.models.load_model("/Users/girijeshs/Downloads/Brave/VGG16_final.keras")

# List all Conv2D layers
for layer in model.layers:
    if 'conv' in layer.name.lower():
        print(f"Conv Layer: {layer.name}")
```

2. **Manually specify the last conv layer** in `app.py`:

Instead of auto-detection, you can hardcode it:

```python
# Around line 164 in app.py, replace:
last_conv_layer_name = get_last_conv_layer_name(model)

# With (use your actual layer name):
last_conv_layer_name = "block5_conv3"  # For VGG16
# or
last_conv_layer_name = "conv2d_12"  # Example generic name
```

3. **Common VGG16 last conv layer names**:
   - `block5_conv3` (standard VGG16)
   - `block5_conv2`
   - `conv2d` (if using sequential model)

## Testing Your Configuration

After updating `app.py`, test it:

```bash
# Start the server
python app.py

# You should see:
# Model loaded successfully from /Users/girijeshs/Downloads/Brave/VGG16_final.keras
# Model input shape: (None, 224, 224, 3)
# Model output shape: (None, 4)  # <-- Number should match your CLASS_LABELS
```

## Quick Test Script

Save this as `test_model.py` and run it:

```python
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model("/Users/girijeshs/Downloads/Brave/VGG16_final.keras")

print("="*50)
print("MODEL INFORMATION")
print("="*50)
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")
print(f"Number of classes: {model.output_shape[-1]}")
print("\nModel Architecture:")
print("-"*50)

# Show all layers
for i, layer in enumerate(model.layers):
    print(f"{i}: {layer.name} - {layer.__class__.__name__}")

print("\nConvolutional Layers:")
print("-"*50)
conv_layers = [layer.name for layer in model.layers if 'Conv' in layer.__class__.__name__]
for conv in conv_layers:
    print(f"  - {conv}")

if conv_layers:
    print(f"\nLast Conv Layer: {conv_layers[-1]}")
    print("👆 Use this name in app.py if Grad-CAM fails!")

print("="*50)
```

Run it:
```bash
python test_model.py
```

This will show you exactly what your model has and what to configure!
