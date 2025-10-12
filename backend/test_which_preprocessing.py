"""
Critical Diagnostic: Determine which preprocessing your model was trained with.

This script will test both preprocessing methods and tell you which one your model expects.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from PIL import Image
import sys

MODEL_PATH = "/Users/girijeshs/Desktop/chrome/Xception_95pct_model.keras"

print("="*80)
print("🔬 PREPROCESSING DIAGNOSTIC TEST")
print("="*80)
print("\nThis will determine which preprocessing your model was trained with.\n")

# Load model
print("Loading model...")
model = keras.models.load_model(MODEL_PATH)
print(f"✅ Model loaded: {model.output_shape[-1]} classes\n")

# Create a test image (or load one if provided)
if len(sys.argv) > 1:
    print(f"Loading test image: {sys.argv[1]}")
    test_image = Image.open(sys.argv[1])
else:
    print("Creating random test image (provide a real MRI image for better results)")
    test_array = np.random.randint(100, 200, (299, 299, 3), dtype=np.uint8)
    test_image = Image.fromarray(test_array)

if test_image.mode != 'RGB':
    test_image = test_image.convert('RGB')
test_image = test_image.resize((299, 299))

# Prepare image as numpy array
img_base = np.array(test_image, dtype=np.float32)

print("\n" + "─"*80)
print("METHOD 1: ImageDataGenerator style (rescale=1./255)")
print("─"*80)

# Method 1: Simple rescale to [0, 1]
img_method1 = img_base.copy()
img_method1 = img_method1 / 255.0
img_method1 = np.expand_dims(img_method1, axis=0)

print(f"Input range: [{img_method1.min():.3f}, {img_method1.max():.3f}]")
preds1 = model.predict(img_method1, verbose=0)
print(f"Predictions: {preds1[0]}")
print(f"Max probability: {preds1[0].max():.4f}")
print(f"Entropy (lower = more confident): {-np.sum(preds1[0] * np.log(preds1[0] + 1e-10)):.4f}")

print("\n" + "─"*80)
print("METHOD 2: Xception preprocess_input (scale to [-1, 1])")
print("─"*80)

# Method 2: Xception preprocessing
img_method2 = img_base.copy()
img_method2 = np.expand_dims(img_method2, axis=0)
img_method2 = xception_preprocess(img_method2)

print(f"Input range: [{img_method2.min():.3f}, {img_method2.max():.3f}]")
preds2 = model.predict(img_method2, verbose=0)
print(f"Predictions: {preds2[0]}")
print(f"Max probability: {preds2[0].max():.4f}")
print(f"Entropy (lower = more confident): {-np.sum(preds2[0] * np.log(preds2[0] + 1e-10)):.4f}")

print("\n" + "="*80)
print("📊 ANALYSIS")
print("="*80)

# Compare confidence levels
conf1 = preds1[0].max()
conf2 = preds2[0].max()

print(f"\nMethod 1 (rescale 1./255) confidence: {conf1:.4f}")
print(f"Method 2 (xception preprocessing) confidence: {conf2:.4f}")

# Determine which is better
diff = abs(conf1 - conf2)

if diff < 0.05:
    print("\n⚠️  INCONCLUSIVE: Both methods give similar confidence.")
    print("Try running this script with a real brain tumor MRI image:")
    print(f"  python {sys.argv[0]} path/to/mri_scan.jpg")
elif conf1 > conf2:
    print(f"\n✅ RESULT: Your model expects METHOD 1 (rescale=1./255)")
    print(f"   Confidence difference: {diff:.4f}")
    print("\n🔧 FIX NEEDED in app.py:")
    print("   Change preprocessing to use simple rescaling:")
    print("   img_array = img_array.astype('float32') / 255.0")
    print("\n   ❌ REMOVE the xception_preprocess() line")
else:
    print(f"\n✅ RESULT: Your model expects METHOD 2 (xception preprocessing)")
    print(f"   Confidence difference: {diff:.4f}")
    print("\n✅ Your current app.py is CORRECT!")
    print("   Keep using: xception_preprocess(img_array)")

print("\n" + "="*80)
print("💡 HOW TO VERIFY:")
print("="*80)
print("""
1. Run this script with a REAL brain tumor MRI image
2. The method with HIGHER confidence is the correct one
3. If Method 1 wins → trained with ImageDataGenerator(rescale=1./255)
4. If Method 2 wins → trained with xception.preprocess_input()

For most accurate results, use a tumor image you know your model should
predict correctly with high confidence.
""")

print("="*80)
