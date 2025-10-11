import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
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

# Define class labels - UPDATE THESE BASED ON YOUR MODEL'S CLASSES
# Common brain tumor classification:
# If your model has 2 classes: {0: "No Tumor", 1: "Tumor"}
# If your model has 4 classes: {0: "Glioma", 1: "Meningioma", 2: "Pituitary", 3: "No Tumor"}
# Adjust based on how your model was trained!
CLASS_LABELS = {
    0: "Glioma Tumor",
    1: "Meningioma Tumor", 
    2: "Pituitary Tumor",
    3: "No Tumor"
}

# If you're unsure about your model's classes, check the output shape
# The number of output neurons = number of classes

def preprocess_image(image, target_size=(299, 299)):
    """
    Preprocess the uploaded image for model prediction.
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size
    image = image.resize(target_size)
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Normalize pixel values (adjust based on your model's training preprocessing)
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate Grad-CAM heatmap for the given image and model.
    Enhanced version with better gradient computation and robust error handling.
    """
    # Create a model that maps the input image to the activations of the last conv layer
    # and the output predictions
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        print(f"Layer '{last_conv_layer_name}' not found in model")
        available_layers = [layer.name for layer in model.layers]
        print(f"Available layers: {available_layers}")
        raise
    
    # Handle both single and multiple inputs/outputs
    model_inputs = model.input
    if isinstance(model_inputs, list):
        model_inputs = model_inputs[0] if len(model_inputs) == 1 else model_inputs
    
    model_outputs = model.output
    if isinstance(model_outputs, list):
        print(f"Model has multiple outputs: {len(model_outputs)}")
        model_outputs = model_outputs[-1]  # Use last output if multiple
        print(f"Using output: {model_outputs}")
    
    grad_model = keras.models.Model(
        inputs=model_inputs,
        outputs=[last_conv_layer.output, model_outputs]
    )
    
    print(f"Grad model created - Input: {grad_model.input_shape}, Outputs: {[o.shape for o in grad_model.outputs]}")
    
    # Compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        # Ensure preds is a tensor, not a list
        if isinstance(preds, list):
            preds = preds[0] if len(preds) == 1 else preds[-1]
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        
        # Ensure pred_index is an integer
        if isinstance(pred_index, (np.integer, int)):
            pred_index = int(pred_index)
        else:
            pred_index = int(pred_index.numpy())
        
        class_channel = preds[:, pred_index]
    
    # Gradient of the output neuron with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Check if gradients are valid
    if grads is None:
        print("Warning: Gradients are None, cannot generate Grad-CAM")
        raise ValueError("Gradient computation failed")
    
    # Vector of mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel in the feature map array by "how important this channel is"
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize the heatmap between 0 & 1 for visualization
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    
    if max_val > 0:
        heatmap = heatmap / max_val
    
    heatmap_np = heatmap.numpy()
    print(f"Heatmap generated - Shape: {heatmap_np.shape}, Min: {heatmap_np.min():.3f}, Max: {heatmap_np.max():.3f}")
    
    return heatmap_np

def create_gradcam_overlay(original_image, heatmap, alpha=0.7):
    """
    Create an overlay of the Grad-CAM heatmap on the original image.
    Enhanced version with MAXIMUM visibility - AGGRESSIVE FIX.
    """
    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, (original_image.width, original_image.height))
    
    # AGGRESSIVE normalization - ensure we have variation
    heatmap_resized = np.maximum(heatmap_resized, 0)
    
    # Apply power transformation to increase contrast
    heatmap_resized = np.power(heatmap_resized, 0.5)  # Square root for better visibility
    
    if heatmap_resized.max() > 0:
        heatmap_resized = heatmap_resized / heatmap_resized.max()
    
    print(f"DEBUG: Heatmap after processing - Min: {heatmap_resized.min():.4f}, Max: {heatmap_resized.max():.4f}, Mean: {heatmap_resized.mean():.4f}")
    
    # Convert heatmap to RGB colormap with FULL intensity
    heatmap_colored = np.uint8(255 * heatmap_resized)
    
    # Apply JET colormap for maximum visibility
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    
    print(f"DEBUG: Heatmap colored - Shape: {heatmap_colored.shape}, Range: {heatmap_colored.min()} to {heatmap_colored.max()}")
    
    # Convert PIL image to numpy array
    img_array = np.array(original_image)
    
    # Ensure image is in RGB format
    if len(img_array.shape) == 2:  # Grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    print(f"DEBUG: Original image - Shape: {img_bgr.shape}, Range: {img_bgr.min()} to {img_bgr.max()}")
    
    # AGGRESSIVE overlay with higher alpha for maximum visibility
    superimposed_img = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)
    
    print(f"DEBUG: Superimposed - Shape: {superimposed_img.shape}, Range: {superimposed_img.min()} to {superimposed_img.max()}")
    
    # Convert back to RGB for final output
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    print(f"✅ Grad-CAM overlay created - Alpha: {alpha}, Heatmap range: {heatmap_resized.min():.3f} to {heatmap_resized.max():.3f}")
    
    return superimposed_img

def get_last_conv_layer_name(model):
    """
    Automatically find the last convolutional layer in the model.
    Supports both Conv2D (VGG, ResNet) and SeparableConv2D (Xception, MobileNet).
    """
    # Manual override - uncomment and set your layer name if auto-detection fails
    # MANUAL_LAYER_NAME = "block5_conv3"  # For VGG16
    # MANUAL_LAYER_NAME = "block14_sepconv2"  # For Xception
    # if MANUAL_LAYER_NAME:
    #     print(f"Using manual layer override: {MANUAL_LAYER_NAME}")
    #     return MANUAL_LAYER_NAME
    
    conv_layers = []
    for layer in model.layers:
        # Check if it's a Conv2D or SeparableConv2D layer
        if isinstance(layer, (keras.layers.Conv2D, keras.layers.SeparableConv2D)):
            conv_layers.append(layer.name)
    
    if not conv_layers:
        # If no convolutional layers found, check nested models (Functional API)
        print("Searching nested models for convolutional layers...")
        for layer in model.layers:
            if hasattr(layer, 'layers'):  # Nested model
                for sublayer in layer.layers:
                    if isinstance(sublayer, (keras.layers.Conv2D, keras.layers.SeparableConv2D)):
                        conv_layers.append(sublayer.name)
    
    if not conv_layers:
        # If still no Conv2D layer found, raise an error
        print("No convolutional layers found in model!")
        print("Available layer types:")
        for layer in model.layers:
            print(f"  - {layer.name}: {layer.__class__.__name__}")
        raise ValueError("No convolutional layer found in the model")
    
    last_conv = conv_layers[-1]
    print(f"Found {len(conv_layers)} conv layers. Using last one: {last_conv}")
    return last_conv

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
        predictions = model.predict(img_array)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        
        # Get prediction label
        prediction_label = CLASS_LABELS.get(predicted_class, f"Class {predicted_class}")
        
        # Get all class probabilities for detailed response
        all_probabilities = {}
        for class_idx, class_name in CLASS_LABELS.items():
            if class_idx < len(predictions[0]):
                all_probabilities[class_name] = float(predictions[0][class_idx])
        
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
