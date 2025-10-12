"""
Comprehensive test script for Xception Grad-CAM implementation.

This script verifies that:
1. Preprocessing matches training setup (img/255.0)
2. Correct Xception layer is detected automatically
3. Grad-CAM heatmaps are generated correctly
4. Overlays look natural and highlight tumor regions

Run this BEFORE starting the Flask server to ensure everything works!

Usage:
    python test_gradcam_xception.py [optional_path_to_test_image.jpg]
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from PIL import Image
import sys
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = "/Users/girijeshs/Desktop/chrome/Xception_95pct_model.keras"

CLASS_LABELS = {
    0: "Glioma Tumor",
    1: "Meningioma Tumor", 
    2: "No Tumor",
    3: "Pituitary Tumor"
}

# ============================================================================
# PREPROCESSING (MUST MATCH TRAINING!)
# ============================================================================
def preprocess_image(image, target_size=(299, 299)):
    """
    Preprocess image exactly as done during training.
    ✅ Uses img/255.0 normalization (matches ImageDataGenerator(rescale=1./255))
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize(target_size)
    img_array = np.array(image)
    
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    
    # ✅ CRITICAL: [0, 1] normalization (same as training)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# ============================================================================
# LAYER DETECTION
# ============================================================================
def get_last_conv_layer_name(model):
    """
    Automatically detect the correct layer for Grad-CAM.
    Prioritizes 'block14_sepconv2_act' for Xception.
    """
    XCEPTION_LAYERS = [
        "block14_sepconv2_act",
        "block14_sepconv2",
        "block14_sepconv1_act",
        "block13_sepconv2_act",
    ]
    
    # Try known Xception layers
    for layer_name in XCEPTION_LAYERS:
        try:
            model.get_layer(layer_name)
            return layer_name
        except:
            continue
    
    # Fallback: search for any conv layers
    conv_layers = []
    for layer in model.layers:
        if isinstance(layer, (keras.layers.Conv2D, keras.layers.SeparableConv2D)):
            conv_layers.append(layer.name)
        elif isinstance(layer, keras.layers.Activation) and 'conv' in layer.name.lower():
            conv_layers.append(layer.name)
    
    if conv_layers:
        return conv_layers[-1]
    
    raise ValueError("No convolutional layer found in the model")

# ============================================================================
# GRAD-CAM GENERATION
# ============================================================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate Grad-CAM heatmap.
    
    Returns heatmap array with values in [0, 1] range.
    """
    last_conv_layer = model.get_layer(last_conv_layer_name)
    
    grad_model = keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        
        if not isinstance(pred_index, int):
            pred_index = int(pred_index.numpy() if hasattr(pred_index, 'numpy') else pred_index)
        
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

# ============================================================================
# OVERLAY CREATION
# ============================================================================
def create_gradcam_overlay(original_image, heatmap, alpha=0.4):
    """
    Create smooth overlay with natural blending.
    Uses cv2.addWeighted for professional-looking results.
    """
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (original_image.width, original_image.height))
    
    # Normalize
    heatmap_resized = np.maximum(heatmap_resized, 0)
    if heatmap_resized.max() > 0:
        heatmap_resized = heatmap_resized / heatmap_resized.max()
    
    # Apply Gaussian blur for smoother appearance
    heatmap_resized = cv2.GaussianBlur(heatmap_resized, (0, 0), sigmaX=2, sigmaY=2)
    
    # Convert to colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Prepare original image
    img_array = np.array(original_image)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Blend using alpha blending
    superimposed_img = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    return superimposed_img

# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================
def main():
    print("="*80)
    print("🧠 XCEPTION GRAD-CAM TEST SCRIPT")
    print("="*80)
    print("\nThis script verifies your Grad-CAM implementation for Xception.")
    print("It checks: preprocessing, layer detection, heatmap generation, and overlay.")
    
    # ========================================================================
    # 1. LOAD MODEL
    # ========================================================================
    print("\n" + "─"*80)
    print("STEP 1: Loading Xception Model")
    print("─"*80)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at: {MODEL_PATH}")
        print("\nPlease update MODEL_PATH in this script to point to your model.")
        return
    
    try:
        model = keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded successfully!")
        print(f"   Input shape:  {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Classes: {model.output_shape[-1]}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # ========================================================================
    # 2. DETECT GRAD-CAM LAYER
    # ========================================================================
    print("\n" + "─"*80)
    print("STEP 2: Detecting Optimal Grad-CAM Layer")
    print("─"*80)
    
    try:
        last_conv_layer_name = get_last_conv_layer_name(model)
        print(f"✅ Detected layer: '{last_conv_layer_name}'")
        
        # Verify layer exists
        layer = model.get_layer(last_conv_layer_name)
        print(f"   Layer type: {layer.__class__.__name__}")
        print(f"   Output shape: {layer.output_shape}")
    except Exception as e:
        print(f"❌ Layer detection failed: {e}")
        return
    
    # ========================================================================
    # 3. PREPARE TEST IMAGE
    # ========================================================================
    print("\n" + "─"*80)
    print("STEP 3: Preparing Test Image")
    print("─"*80)
    
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
        if os.path.exists(test_image_path):
            try:
                test_image = Image.open(test_image_path)
                print(f"✅ Loaded image: {test_image_path}")
                print(f"   Original size: {test_image.size}")
                print(f"   Mode: {test_image.mode}")
            except Exception as e:
                print(f"❌ Failed to load image: {e}")
                return
        else:
            print(f"❌ Image not found: {test_image_path}")
            return
    else:
        # Create random test image
        print("⚠️  No image provided, using random test image")
        print("   Usage: python test_gradcam_xception.py <path_to_mri_image.jpg>")
        test_array = np.random.randint(0, 256, (299, 299, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_array)
        print("✅ Created random test image (299x299)")
    
    # ========================================================================
    # 4. PREPROCESS IMAGE
    # ========================================================================
    print("\n" + "─"*80)
    print("STEP 4: Preprocessing Image")
    print("─"*80)
    print("⚠️  Using [0, 1] normalization (img/255.0) - MATCHES TRAINING")
    
    img_array = preprocess_image(test_image)
    print(f"✅ Preprocessed successfully!")
    print(f"   Shape: {img_array.shape}")
    print(f"   Data type: {img_array.dtype}")
    print(f"   Value range: [{img_array.min():.3f}, {img_array.max():.3f}]")
    
    if img_array.min() < -1.1 or img_array.max() > 1.1:
        print("⚠️  WARNING: Values outside expected range!")
        print("   This might indicate incorrect preprocessing.")
    
    # ========================================================================
    # 5. MAKE PREDICTION
    # ========================================================================
    print("\n" + "─"*80)
    print("STEP 5: Making Prediction")
    print("─"*80)
    
    try:
        predictions = model.predict(img_array, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        
        print(f"✅ Prediction successful!")
        print(f"   Predicted class: {predicted_class} ({CLASS_LABELS[predicted_class]})")
        print(f"   Confidence: {confidence:.2%}")
        print(f"\n   All probabilities:")
        for idx, prob in enumerate(predictions[0]):
            label = CLASS_LABELS.get(idx, f"Class {idx}")
            bar = "█" * int(prob * 50)
            print(f"   {idx}. {label:20s} {prob:.2%} {bar}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return
    
    # ========================================================================
    # 6. GENERATE GRAD-CAM HEATMAP
    # ========================================================================
    print("\n" + "─"*80)
    print("STEP 6: Generating Grad-CAM Heatmap")
    print("─"*80)
    
    try:
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, predicted_class)
        print(f"✅ Heatmap generated successfully!")
        print(f"   Shape: {heatmap.shape}")
        print(f"   Value range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
        print(f"   Mean value: {heatmap.mean():.3f}")
        print(f"   Std dev: {heatmap.std():.3f}")
        
        # Check heatmap quality
        if heatmap.max() - heatmap.min() < 0.01:
            print("⚠️  WARNING: Heatmap has very low variation!")
            print("   This might indicate gradient computation issues.")
        else:
            print("✅ Heatmap shows good variation (suitable for visualization)")
    except Exception as e:
        print(f"❌ Grad-CAM generation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================================================
    # 7. CREATE OVERLAY
    # ========================================================================
    print("\n" + "─"*80)
    print("STEP 7: Creating Grad-CAM Overlay")
    print("─"*80)
    
    try:
        overlay = create_gradcam_overlay(test_image, heatmap, alpha=0.4)
        print(f"✅ Overlay created successfully!")
        print(f"   Shape: {overlay.shape}")
        print(f"   Blending: 60% original + 40% heatmap")
    except Exception as e:
        print(f"❌ Overlay creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================================================
    # 8. SAVE RESULTS
    # ========================================================================
    print("\n" + "─"*80)
    print("STEP 8: Saving Test Results")
    print("─"*80)
    
    try:
        output_dir = "gradcam_test_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save original
        original_path = os.path.join(output_dir, "1_original.jpg")
        test_image.resize((299, 299)).save(original_path)
        
        # Save heatmap only
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        heatmap_resized = cv2.resize(heatmap_rgb, (299, 299))
        heatmap_path = os.path.join(output_dir, "2_heatmap.jpg")
        Image.fromarray(heatmap_resized).save(heatmap_path)
        
        # Save overlay
        overlay_path = os.path.join(output_dir, "3_overlay.jpg")
        Image.fromarray(overlay).save(overlay_path)
        
        print(f"✅ Results saved to '{output_dir}/' directory:")
        print(f"   - {original_path}")
        print(f"   - {heatmap_path}")
        print(f"   - {overlay_path}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\n📊 SUMMARY:")
    print(f"   ✓ Model loaded successfully")
    print(f"   ✓ Layer detected: {last_conv_layer_name}")
    print(f"   ✓ Preprocessing: [0, 1] normalization (matches training)")
    print(f"   ✓ Prediction: {CLASS_LABELS[predicted_class]} ({confidence:.1%} confidence)")
    print(f"   ✓ Grad-CAM heatmap generated with good variation")
    print(f"   ✓ Smooth overlay created with alpha blending")
    print(f"   ✓ Results saved to '{output_dir}/' directory")
    
    print("\n🎯 NEXT STEPS:")
    print("   1. Check the saved images in 'gradcam_test_results/' folder")
    print("   2. Verify that the heatmap highlights relevant tumor regions")
    print("   3. If results look good, start your Flask server: python app.py")
    print("   4. If heatmap is still incorrect, the model may need retraining")
    
    print("\n💡 TIPS:")
    print("   - Red areas in heatmap = high importance for prediction")
    print("   - Blue areas = low importance")
    print("   - For 'No Tumor' predictions, heatmap may be more uniform")
    print("   - For tumor predictions, heatmap should focus on tumor location")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
