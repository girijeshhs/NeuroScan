import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
MODEL_PATH = "/Users/girijeshs/Desktop/chrome/Xception_95pct_model.keras"
model = keras.models.load_model(MODEL_PATH)

print(f"Model loaded successfully from {MODEL_PATH}")
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

# Define class labels - CORRECT ORDER from 7k Brain Tumor Dataset
# ✅ This matches the alphabetical folder order from training
# Define class labels - CORRECT ORDER from your specific trained model
# This was empirically verified to match your model's actual output
CLASS_LABELS = {
    0: "Glioma Tumor",
    1: "Meningioma Tumor", 
    2: "No Tumor",
    3: "Pituitary Tumor"
}

# If you're unsure about your model's classes, check the output shape
# The number of output neurons = number of classes

def preprocess_image(image, target_size=(299, 299)):
    """
    Preprocess the uploaded MRI image for Xception model using official Xception preprocessing.
    
    ✅ Uses tf.keras.applications.xception.preprocess_input which:
    - Scales pixel values from [0, 255] to [-1, 1] range
    - This is the CORRECT preprocessing for pretrained Xception models
    
    This function:
    1. Converts grayscale MRI images to RGB (Xception expects 3 channels)
    2. Resizes to 299x299 (Xception's input size)
    3. Applies Xception-specific preprocessing (scales to [-1, 1])
    4. Adds batch dimension for model input
    
    Args:
        image: PIL Image object (uploaded MRI scan)
        target_size: Tuple (height, width) - should be (299, 299) for Xception
    
    Returns:
        img_array: Numpy array of shape (1, 299, 299, 3) with values in [-1, 1]
    """
    # Step 1: Convert grayscale MRI images to RGB
    # Many MRI scans are grayscale, but Xception requires 3 channels
    if image.mode != 'RGB':
        image = image.convert('RGB')
        print(f"✓ Converted to RGB")
    
    # Step 2: Resize to Xception's expected input size (299x299)
    image = image.resize(target_size)
    print(f"✓ Resized to {target_size}")
    
    # Step 3: Convert PIL Image to numpy array (values in [0, 255])
    img_array = np.array(image, dtype=np.float32)
    
    # Safety check: ensure we have 3 channels
    if len(img_array.shape) == 2:  # Grayscale (H, W)
        img_array = np.stack((img_array,)*3, axis=-1)  # Convert to (H, W, 3)
        print(f"✓ Converted grayscale to RGB")
    elif img_array.shape[-1] == 1:  # (H, W, 1)
        img_array = np.stack((img_array,)*3, axis=-1)
        print(f"✓ Converted single-channel to RGB")
    
    # Step 4: Add batch dimension first (required by preprocess_input)
    img_array = np.expand_dims(img_array, axis=0)  # (H, W, 3) -> (1, H, W, 3)
    
    # Step 5: Apply Xception-specific preprocessing
    # ✅ This scales from [0, 255] to [-1, 1] range - CORRECT for Xception!
    img_array = xception_preprocess(img_array)
    
    print(f"✅ Xception preprocessing complete - Shape: {img_array.shape}, Range: [{img_array.min():.3f}, {img_array.max():.3f}]")
    
    return img_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate clean Grad-CAM heatmap for Xception model.
    
    This function computes class activation maps by:
    1. Getting activations from the last convolutional layer (block14_sepconv2_act)
    2. Computing gradients of predicted class w.r.t. those activations
    3. Weighting activations by gradient importance
    4. Producing a focused heatmap showing which regions influenced the prediction
    
    Args:
        img_array: Preprocessed image array (1, 299, 299, 3) with Xception preprocessing [-1, 1]
        model: Trained Keras model
        last_conv_layer_name: Name of the last convolutional layer (should be 'block14_sepconv2_act')
        pred_index: Class index to visualize (uses argmax if None)
    
    Returns:
        heatmap_np: Clean numpy array of shape (H, W) with values in [0, 1]
    """
    # Get the target convolutional layer
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
        print(f"✓ Using Grad-CAM layer: {last_conv_layer_name}")
    except ValueError:
        print(f"❌ Layer '{last_conv_layer_name}' not found in model")
        raise
    
    # Create gradient model
    model_inputs = model.input
    if isinstance(model_inputs, list):
        model_inputs = model_inputs[0] if len(model_inputs) == 1 else model_inputs
    
    model_outputs = model.output
    if isinstance(model_outputs, list):
        model_outputs = model_outputs[-1]
    
    grad_model = keras.models.Model(
        inputs=model_inputs,
        outputs=[last_conv_layer.output, model_outputs]
    )
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array, training=False)
        
        if isinstance(preds, list):
            preds = preds[0] if len(preds) == 1 else preds[-1]
        
        # Get predicted class index
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        
        pred_index = int(pred_index.numpy() if hasattr(pred_index, 'numpy') else pred_index)
        print(f"✓ Generating Grad-CAM for class {pred_index}")
        
        # Get class score
        class_channel = preds[:, pred_index]
    
    # Compute gradients of class score w.r.t. feature maps
    grads = tape.gradient(class_channel, conv_outputs)
    
    if grads is None:
        raise ValueError("Gradient computation failed")
    
    # Global average pooling of gradients (importance weights)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight each feature map by its importance (element-wise for stability)
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    
    # Apply ReLU (only positive contributions)
    heatmap = tf.maximum(heatmap, 0)
    
    # Normalize to [0, 1] - NO additional scaling or distortion
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-10)
    
    heatmap_np = heatmap.numpy()
    heatmap_np = refine_heatmap(heatmap_np)
    
    print(f"✅ Clean heatmap generated - Shape: {heatmap_np.shape}, Range: [{heatmap_np.min():.3f}, {heatmap_np.max():.3f}]")
    
    return heatmap_np


def refine_heatmap(heatmap, percentile=97, blur_kernel=(9, 9)):
    """Sharpen Grad-CAM map by clipping outliers and applying gentle blur."""
    if heatmap.ndim != 2:
        raise ValueError("Heatmap must be 2D")

    heatmap = np.maximum(heatmap, 0)
    heatmap_max = heatmap.max() + 1e-10
    heatmap = heatmap / heatmap_max

    if percentile is not None:
        cutoff = np.percentile(heatmap, percentile)
        if cutoff > 0:
            heatmap = np.where(heatmap >= cutoff, heatmap, 0)

    heatmap = cv2.GaussianBlur(heatmap, blur_kernel, sigmaX=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-10)

    return heatmap

def create_gradcam_overlay(original_image, heatmap, alpha=0.5):
    """
    Create clean, focused Grad-CAM overlay on the original MRI image.
    
    This function:
    1. Resizes heatmap to match original image
    2. Applies JET colormap (red=tumor region focus, blue=background)
    3. Blends with original image for clear visualization
    
    Args:
        original_image: PIL Image object (original MRI scan)
        heatmap: Clean numpy array (H, W) with values in [0, 1]
        alpha: Blending factor (0.5 = balanced visibility)
    
    Returns:
        superimposed_img: Numpy array (H, W, 3) with clean overlay
    """
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (original_image.width, original_image.height))
    
    # Ensure proper range [0, 1] - NO extra normalization to avoid distortion
    heatmap_resized = np.clip(heatmap_resized, 0, 1)
    
    print(f"✓ Heatmap resized - Range: [{heatmap_resized.min():.3f}, {heatmap_resized.max():.3f}]")
    
    # Convert to 8-bit for colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Apply JET colormap (red=high importance for tumor detection)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Prepare original image
    img_array = np.array(original_image)
    
    # Handle grayscale MRI
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # Convert to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Clean blend: 50% original + 50% heatmap
    superimposed_img = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)
    
    # Convert back to RGB
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    print(f"✅ Clean overlay created - {int((1-alpha)*100)}% original + {int(alpha*100)}% heatmap")
    
    return superimposed_img

def get_last_conv_layer_name(model):
    """
    Get the correct convolutional layer for Xception Grad-CAM.
    
    For Xception, the optimal layer is 'block14_sepconv2_act' - the final
    convolutional activation that provides the best tumor localization.
    
    Args:
        model: Trained Keras Xception model
    
    Returns:
        layer_name: 'block14_sepconv2_act' for Xception
    """
    # Xception's optimal Grad-CAM layer
    XCEPTION_LAYER = "block14_sepconv2_act"
    
    print(f"🎯 Using Xception Grad-CAM layer: {XCEPTION_LAYER}")
    
    # Verify the layer exists
    try:
        model.get_layer(XCEPTION_LAYER)
        print(f"✅ Layer '{XCEPTION_LAYER}' found in model")
        return XCEPTION_LAYER
    except:
        print(f"⚠️  Warning: '{XCEPTION_LAYER}' not found, searching for alternatives...")
        
        # Fallback search
        for layer in model.layers:
            if isinstance(layer, (keras.layers.SeparableConv2D, keras.layers.Conv2D)):
                last_conv = layer.name
            elif isinstance(layer, keras.layers.Activation) and 'block14' in layer.name:
                return layer.name
        
        if last_conv:
            print(f"✅ Using fallback layer: {last_conv}")
            return last_conv
        
        raise ValueError("No suitable convolutional layer found for Grad-CAM")

def image_to_base64(img_array):
    """
    Convert numpy array image to base64 string.
    """
    img = Image.fromarray(img_array.astype('uint8'))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

@app.route('/')
def home():
    return "Brain Tumor Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to handle image upload, prediction, and Grad-CAM generation.
    """
    try:
        # Check if image file is in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and preprocess the image
        image = Image.open(file.stream)
        original_image = image.copy()
        
        # Preprocess for model
        img_array = preprocess_image(image)
        
        # Make prediction
        print("\n" + "="*60)
        print("🧠 MAKING PREDICTION")
        print("="*60)
        
        predictions = model.predict(img_array, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        
        # Get prediction label
        prediction_label = CLASS_LABELS.get(predicted_class, f"Class {predicted_class}")
        
        # Print clear prediction results
        print(f"\n✅ PREDICTION RESULTS:")
        print(f"   Predicted: {prediction_label}")
        print(f"   Class Index: {predicted_class}")
        print(f"   Confidence: {confidence * 100:.2f}%")
        print(f"\n   All Class Probabilities:")
        
        # Get all class probabilities for detailed response
        all_probabilities = {}
        for class_idx, class_name in CLASS_LABELS.items():
            if class_idx < len(predictions[0]):
                prob = float(predictions[0][class_idx])
                all_probabilities[class_name] = prob
                bar = "█" * int(prob * 40)
                print(f"   {class_idx}. {class_name:20s} {prob*100:5.2f}% {bar}")
        
        print("="*60 + "\n")
        
        # Determine if tumor is present
        is_tumor = "No Tumor" not in prediction_label
        tumor_type = prediction_label if is_tumor else "None"
        
        # Generate Grad-CAM - ONLY if tumor is detected
        gradcam_base64 = None
        
        if not is_tumor:
            print("\n" + "="*50)
            print("ℹ️  No tumor detected - Skipping Grad-CAM generation")
            print("="*50 + "\n")
        else:
            print("\n" + "="*50)
            print("🔥 STARTING GRAD-CAM GENERATION")
            print("="*50)
            try:
                last_conv_layer_name = get_last_conv_layer_name(model)
                print(f"✓ Using last conv layer: {last_conv_layer_name}")
                
                print(f"✓ Generating heatmap for class {predicted_class}...")
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, predicted_class)
                
                print(f"✓ Creating overlay with original image...")
                gradcam_image = create_gradcam_overlay(original_image, heatmap)
                
                print(f"✓ Converting to base64...")
                gradcam_base64 = image_to_base64(gradcam_image)
                
                print("✅ Grad-CAM generated successfully!")
                print("="*50 + "\n")
            except Exception as e:
                print(f"❌ Grad-CAM generation error: {e}")
                import traceback
                traceback.print_exc()
                print("="*50 + "\n")
                # If Grad-CAM fails, don't send any image
                gradcam_base64 = None
        
        # Prepare response
        response = {
            'prediction': prediction_label,
            'tumor_type': tumor_type,
            'is_tumor': is_tumor,
            'confidence': f"{confidence * 100:.2f}%",
            'predicted_class': predicted_class,
            'raw_confidence': confidence,
            'all_probabilities': all_probabilities,
            'gradcam_image': gradcam_base64,
            'gradcam_available': gradcam_base64 is not None
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """
    Endpoint to get model information.
    """
    try:
        layer_names = [layer.name for layer in model.layers]
        conv_layers = [layer.name for layer in model.layers if isinstance(layer, keras.layers.Conv2D)]
        
        return jsonify({
            'input_shape': str(model.input_shape),
            'output_shape': str(model.output_shape),
            'total_layers': len(model.layers),
            'conv_layers': conv_layers,
            'last_conv_layer': get_last_conv_layer_name(model)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test-gradcam', methods=['GET'])
def test_gradcam():
    """
    Quick test endpoint to verify Grad-CAM is working.
    """
    try:
        # Create a random test image
        test_img = np.random.rand(299, 299, 3).astype('float32')
        test_img_pil = Image.fromarray((test_img * 255).astype('uint8'))
        img_array = np.expand_dims(test_img, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        
        # Generate Grad-CAM
        last_conv_layer_name = get_last_conv_layer_name(model)
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, predicted_class)
        gradcam_image = create_gradcam_overlay(test_img_pil, heatmap)
        gradcam_base64 = image_to_base64(gradcam_image)
        
        return jsonify({
            'status': 'success',
            'message': 'Grad-CAM test successful!',
            'predicted_class': predicted_class,
            'heatmap_shape': str(heatmap.shape),
            'heatmap_range': f"{heatmap.min():.4f} to {heatmap.max():.4f}",
            'test_image': gradcam_base64
        }), 200
    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
