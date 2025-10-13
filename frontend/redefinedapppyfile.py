import os, numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2, base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)

MODEL_PATH = "/Users/girijeshs/Desktop/chrome/Xception_95pct_model.keras"
model = keras.models.load_model(MODEL_PATH)

CLASS_LABELS = {0: "Glioma Tumor", 1: "Meningioma Tumor", 2: "No Tumor", 3: "Pituitary Tumor"}

def preprocess_image(image, target_size=(299, 299)):
    if image.mode != 'RGB': image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32)
    if len(img_array.shape) == 2 or img_array.shape[-1]==1: img_array = np.stack((img_array,)*3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return xception_preprocess(img_array)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = keras.models.Model(inputs=model.input, outputs=[last_conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        if pred_index is None: pred_index = tf.argmax(preds[0])
        class_channel = preds[:, int(pred_index.numpy() if hasattr(pred_index, 'numpy') else pred_index)]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    heatmap = tf.squeeze(conv_outputs[0] @ pooled_grads[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap)+1e-10)
    return heatmap.numpy()

def create_gradcam_overlay(original_image, heatmap, alpha=0.5):
    heatmap = cv2.resize(np.clip(heatmap,0,1),(original_image.width,original_image.height))
    heatmap_colored = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    img_array = np.array(original_image)
    if len(img_array.shape)==2: img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 1-alpha, heatmap_colored, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

def get_last_conv_layer_name(model):
    layer_name = "block14_sepconv2_act"
    try: model.get_layer(layer_name); return layer_name
    except: return next((layer.name for layer in model.layers if isinstance(layer, (keras.layers.SeparableConv2D, keras.layers.Conv2D))), None)

def image_to_base64(img_array):
    buf = BytesIO(); Image.fromarray(img_array.astype('uint8')).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

@app.route('/')
def home(): return "Brain Tumor Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file or file.filename=='': return jsonify({'error':'No file uploaded'}), 400
    image = Image.open(file.stream); original_image = image.copy()
    img_array = preprocess_image(image)
    predictions = model.predict(img_array, verbose=0)
    predicted_class = int(np.argmax(predictions[0])); confidence = float(predictions[0][predicted_class])
    prediction_label = CLASS_LABELS.get(predicted_class, f"Class {predicted_class}")
    is_tumor = prediction_label!="No Tumor"; tumor_type = prediction_label if is_tumor else "None"
    gradcam_base64 = None
    if is_tumor:
        last_conv_layer_name = get_last_conv_layer_name(model)
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, predicted_class)
        gradcam_base64 = image_to_base64(create_gradcam_overlay(original_image, heatmap))
    response = {
        'prediction': prediction_label, 'tumor_type': tumor_type, 'is_tumor': is_tumor,
        'confidence': f"{confidence*100:.2f}%", 'predicted_class': predicted_class,
        'raw_confidence': confidence, 'all_probabilities': {CLASS_LABELS[i]: float(predictions[0][i]) for i in CLASS_LABELS},
        'gradcam_image': gradcam_base64, 'gradcam_available': gradcam_base64 is not None
    }
    return jsonify(response), 200

@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape),
        'total_layers': len(model.layers),
        'conv_layers': [layer.name for layer in model.layers if isinstance(layer, keras.layers.Conv2D)],
        'last_conv_layer': get_last_conv_layer_name(model)
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
