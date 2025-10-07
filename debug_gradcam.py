"""
Debug script to test Grad-CAM generation with a sample image.
This helps identify issues before running the full web app.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from PIL import Image
import sys

# Configuration
MODEL_PATH = "/Users/girijeshs/Downloads/Brave/VGG16_final.keras"

def get_last_conv_layer_name(model):
    """Find the last convolutional layer."""
    conv_layers = []
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            conv_layers.append(layer.name)
    
    if not conv_layers:
        raise ValueError("No convolutional layers found")
    
    return conv_layers[-1]

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap."""
    last_conv_layer = model.get_layer(last_conv_layer_name)
    
    grad_model = keras.models.Model(
        inputs=[model.inputs],
        outputs=[last_conv_layer.output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    if grads is None:
        raise ValueError("Gradient computation failed")
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    
    return heatmap.numpy()

def create_gradcam_overlay(original_image, heatmap, alpha=0.6):
    """Create Grad-CAM overlay."""
    # Get image dimensions
    if isinstance(original_image, Image.Image):
        width, height = original_image.size
        img_array = np.array(original_image)
    else:
        height, width = original_image.shape[:2]
        img_array = original_image
    
    # Resize heatmap
    heatmap_resized = cv2.resize(heatmap, (width, height))
    
    # Normalize
    heatmap_resized = np.maximum(heatmap_resized, 0)
    if heatmap_resized.max() > 0:
        heatmap_resized = heatmap_resized / heatmap_resized.max()
    
    # Apply colormap
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    
    # Convert image to BGR
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Overlay
    superimposed = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)
    superimposed = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
    
    return superimposed, heatmap_colored

def main():
    print("="*70)
    print("🔍 GRAD-CAM DEBUG SCRIPT")
    print("="*70)
    
    # Check if image path provided
    if len(sys.argv) < 2:
        print("\n⚠️  No image provided for testing")
        print("\nUsage: python debug_gradcam.py <path_to_test_image.jpg>")
        print("\nContinuing with a random test image...")
        use_random = True
    else:
        test_image_path = sys.argv[1]
        use_random = False
    
    # Load model
    print("\n📦 Loading model...")
    try:
        model = keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Get model info
    print(f"\n📊 Model Information:")
    print(f"  Input shape:  {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    
    # Find conv layers
    print(f"\n🔧 Searching for convolutional layers...")
    conv_layers = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            conv_layers.append(layer.name)
            print(f"  {i}: {layer.name} - {layer.output_shape}")
    
    if not conv_layers:
        print("❌ No convolutional layers found!")
        return
    
    last_conv = conv_layers[-1]
    print(f"\n✅ Will use last conv layer: {last_conv}")
    
    # Prepare test image
    print(f"\n🖼️  Preparing test image...")
    if use_random:
        # Create random image
        test_image = np.random.rand(224, 224, 3) * 255
        test_image = test_image.astype('uint8')
        test_image_pil = Image.fromarray(test_image)
        print("  Using randomly generated image")
    else:
        try:
            test_image_pil = Image.open(test_image_path)
            print(f"  Loaded: {test_image_path}")
            print(f"  Size: {test_image_pil.size}")
        except Exception as e:
            print(f"❌ Failed to load image: {e}")
            return
    
    # Preprocess
    if test_image_pil.mode != 'RGB':
        test_image_pil = test_image_pil.convert('RGB')
    
    test_image_resized = test_image_pil.resize((224, 224))
    img_array = np.array(test_image_resized).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    print(f"\n🧪 Making prediction...")
    try:
        predictions = model.predict(img_array, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        print(f"  Predicted class: {predicted_class}")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  All predictions: {predictions[0]}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return
    
    # Generate Grad-CAM
    print(f"\n🎨 Generating Grad-CAM heatmap...")
    try:
        heatmap = make_gradcam_heatmap(img_array, model, last_conv, predicted_class)
        print(f"✅ Heatmap generated successfully")
        print(f"  Shape: {heatmap.shape}")
        print(f"  Min value: {heatmap.min():.4f}")
        print(f"  Max value: {heatmap.max():.4f}")
        print(f"  Mean value: {heatmap.mean():.4f}")
        
        # Check if heatmap has variation
        if heatmap.max() - heatmap.min() < 0.01:
            print("⚠️  WARNING: Heatmap has very little variation!")
            print("   This might indicate an issue with gradient computation.")
        
    except Exception as e:
        print(f"❌ Grad-CAM generation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create overlay
    print(f"\n🖌️  Creating Grad-CAM overlay...")
    try:
        overlayed, heatmap_colored = create_gradcam_overlay(test_image_resized, heatmap, alpha=0.6)
        print(f"✅ Overlay created successfully")
        
        # Save results
        output_original = "debug_original.jpg"
        output_heatmap = "debug_heatmap.jpg"
        output_overlay = "debug_overlay.jpg"
        
        test_image_resized.save(output_original)
        Image.fromarray(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)).save(output_heatmap)
        Image.fromarray(overlayed).save(output_overlay)
        
        print(f"\n💾 Saved debug images:")
        print(f"  - {output_original} (original)")
        print(f"  - {output_heatmap} (heatmap only)")
        print(f"  - {output_overlay} (overlay)")
        
    except Exception as e:
        print(f"❌ Overlay creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Summary
    print("\n" + "="*70)
    print("✅ GRAD-CAM TEST COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\n📝 Next steps:")
    print("  1. Check the saved debug images to verify Grad-CAM works")
    print("  2. If the overlay looks good, your Flask app should work")
    print(f"  3. If needed, manually set layer in app.py: last_conv_layer_name = '{last_conv}'")
    print("\n🚀 Ready to run: python app.py")

if __name__ == "__main__":
    main()
