"""
Test script to find correct preprocessing and class label order for Xception model.
Run this with a known tumor image to see which configuration works correctly.

Usage: python test_xception_config.py <path_to_test_mri_image.jpg>
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import sys

MODEL_PATH = "/Users/girijeshs/Desktop/chrome/Xception_95pct_model.keras"

# Two possible class label orders
LABELS_CONFIG_1 = {
    0: "Glioma Tumor",
    1: "Meningioma Tumor", 
    2: "Pituitary Tumor",
    3: "No Tumor"
}

LABELS_CONFIG_2 = {
    0: "Glioma Tumor",
    1: "Meningioma Tumor", 
    2: "No Tumor",
    3: "Pituitary Tumor"
}

def preprocess_method_1(image):
    """Method 1: [0, 1] normalization"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((299, 299))
    img_array = np.array(image).astype('float32') / 255.0
    return np.expand_dims(img_array, axis=0)

def preprocess_method_2(image):
    """Method 2: [-1, 1] Xception standard"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((299, 299))
    img_array = np.array(image).astype('float32')
    img_array = (img_array / 127.5) - 1.0
    return np.expand_dims(img_array, axis=0)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Please provide a test image path!")
        print("Usage: python test_xception_config.py <path_to_test_image.jpg>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print("="*80)
    print("🔬 XCEPTION MODEL CONFIGURATION TEST")
    print("="*80)
    
    # Load model
    print("\n📦 Loading Xception model...")
    try:
        model = keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded!")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        num_classes = model.output_shape[-1]
        print(f"   Number of classes: {num_classes}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Load test image
    print(f"\n📸 Loading test image: {image_path}")
    try:
        image = Image.open(image_path)
        print(f"✅ Image loaded: {image.size}, mode: {image.mode}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        sys.exit(1)
    
    # Test both preprocessing methods
    print("\n" + "="*80)
    print("🧪 TESTING PREPROCESSING METHODS")
    print("="*80)
    
    methods = [
        ("Method 1: [0, 1] normalization", preprocess_method_1),
        ("Method 2: [-1, 1] Xception standard", preprocess_method_2)
    ]
    
    results = []
    
    for method_name, preprocess_func in methods:
        print(f"\n{method_name}")
        print("-"*80)
        
        img_array = preprocess_func(image)
        print(f"Input range: [{img_array.min():.3f}, {img_array.max():.3f}]")
        
        predictions = model.predict(img_array, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        
        print(f"Predicted class index: {predicted_class}")
        print(f"Confidence: {confidence:.1%}")
        print(f"\nAll probabilities:")
        for i in range(num_classes):
            prob = predictions[0][i]
            bar = "█" * int(prob * 50)
            print(f"  Class {i}: {prob:6.1%} {bar}")
        
        results.append({
            'method': method_name,
            'class': predicted_class,
            'confidence': confidence,
            'probs': predictions[0]
        })
    
    # Show class label mappings
    print("\n" + "="*80)
    print("🏷️  CLASS LABEL CONFIGURATIONS")
    print("="*80)
    
    print("\nConfiguration 1 (Current in app.py):")
    for idx, label in LABELS_CONFIG_1.items():
        print(f"  {idx}: {label}")
    
    print("\nConfiguration 2 (Alternative - No Tumor at index 2):")
    for idx, label in LABELS_CONFIG_2.items():
        print(f"  {idx}: {label}")
    
    # Analysis
    print("\n" + "="*80)
    print("📊 ANALYSIS & RECOMMENDATIONS")
    print("="*80)
    
    print("\n1️⃣ Best Preprocessing Method:")
    best_method = max(results, key=lambda x: x['confidence'])
    print(f"   ✅ {best_method['method']}")
    print(f"   Predicted class: {best_method['class']}")
    print(f"   Confidence: {best_method['confidence']:.1%}")
    
    print("\n2️⃣ Interpretation with Config 1:")
    for i, result in enumerate(results, 1):
        label = LABELS_CONFIG_1.get(result['class'], f"Class {result['class']}")
        print(f"   Method {i}: {label} ({result['confidence']:.1%})")
    
    print("\n3️⃣ Interpretation with Config 2:")
    for i, result in enumerate(results, 1):
        label = LABELS_CONFIG_2.get(result['class'], f"Class {result['class']}")
        print(f"   Method {i}: {label} ({result['confidence']:.1%})")
    
    print("\n💡 WHAT TO DO NEXT:")
    print("-"*80)
    print("1. Look at your test image and determine what it SHOULD be classified as")
    print("2. Compare with the predictions above")
    print("3. Choose the preprocessing + label config that matches reality")
    print("4. Update app.py accordingly")
    
    print("\n📝 TO UPDATE app.py:")
    print("-"*80)
    
    if "Method 1" in best_method['method']:
        print("✅ Keep current preprocessing (already using [0, 1])")
    else:
        print("⚠️  Change preprocessing in app.py line ~50 to:")
        print("   img_array = (img_array / 127.5) - 1.0  # [-1, 1] range")
    
    print("\n🔍 Check your training folder structure or code to confirm class order!")
    print("="*80 + "\n")
