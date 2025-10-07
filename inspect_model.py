"""
Comprehensive model inspection script to understand your VGG16 model structure.
This helps identify why Grad-CAM might fail.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

MODEL_PATH = "/Users/girijeshs/Downloads/Brave/VGG16_final.keras"

print("="*70)
print("🔍 COMPREHENSIVE MODEL INSPECTION")
print("="*70)

try:
    # Load model
    print("\n📦 Loading model...")
    model = keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully\n")
    
    # Basic info
    print("📊 BASIC INFORMATION:")
    print("-"*70)
    print(f"Model type: {type(model).__name__}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    
    # Check if input/output are lists
    print(f"\nInput is list: {isinstance(model.input, list)}")
    print(f"Output is list: {isinstance(model.output, list)}")
    
    if isinstance(model.input, list):
        print(f"Number of inputs: {len(model.input)}")
        for i, inp in enumerate(model.input):
            print(f"  Input {i}: {inp.shape}")
    
    if isinstance(model.output, list):
        print(f"Number of outputs: {len(model.output)}")
        for i, out in enumerate(model.output):
            print(f"  Output {i}: {out.shape}")
    
    # Layer analysis
    print(f"\n🔧 LAYER ANALYSIS:")
    print("-"*70)
    print(f"Total layers: {len(model.layers)}")
    
    # Count layer types
    layer_types = {}
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
    
    print("\nLayer type counts:")
    for ltype, count in sorted(layer_types.items()):
        print(f"  {ltype}: {count}")
    
    # Convolutional layers
    print(f"\n🎯 CONVOLUTIONAL LAYERS (Critical for Grad-CAM):")
    print("-"*70)
    conv_layers = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            conv_layers.append((i, layer.name, layer.output_shape))
            print(f"  [{i:3d}] {layer.name:30s} - Output: {layer.output_shape}")
    
    if not conv_layers:
        print("  ⚠️  WARNING: No Conv2D layers found!")
        print("  Grad-CAM requires convolutional layers to work.")
    else:
        last_conv_idx, last_conv_name, last_conv_shape = conv_layers[-1]
        print(f"\n✅ Last Conv Layer: {last_conv_name} (index {last_conv_idx})")
        print(f"   Output shape: {last_conv_shape}")
    
    # Dense/Output layers
    print(f"\n📊 DENSE/OUTPUT LAYERS:")
    print("-"*70)
    for i, layer in enumerate(model.layers):
        if isinstance(layer, (keras.layers.Dense, keras.layers.Activation)):
            print(f"  [{i:3d}] {layer.name:30s} - {layer.__class__.__name__} - Output: {layer.output_shape}")
    
    # Test prediction
    print(f"\n🧪 TEST PREDICTION:")
    print("-"*70)
    test_input = np.random.rand(1, 224, 224, 3).astype('float32')
    print(f"Test input shape: {test_input.shape}")
    
    try:
        predictions = model.predict(test_input, verbose=0)
        print(f"✅ Prediction successful")
        print(f"Prediction type: {type(predictions)}")
        
        if isinstance(predictions, list):
            print(f"Predictions is a LIST with {len(predictions)} elements")
            for i, pred in enumerate(predictions):
                print(f"  Element {i}: shape={pred.shape}, type={type(pred)}")
        else:
            print(f"Prediction shape: {predictions.shape}")
            print(f"Prediction dtype: {predictions.dtype}")
            print(f"Sample values: {predictions[0]}")
            predicted_class = np.argmax(predictions[0])
            print(f"Predicted class: {predicted_class}")
            print(f"Confidence: {predictions[0][predicted_class]:.4f}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Grad-CAM compatibility
    print(f"\n🎨 GRAD-CAM COMPATIBILITY TEST:")
    print("-"*70)
    
    if not conv_layers:
        print("❌ No convolutional layers - Grad-CAM NOT possible")
    else:
        try:
            last_conv_name = conv_layers[-1][1]
            last_conv_layer = model.get_layer(last_conv_name)
            
            # Handle multiple outputs
            model_outputs = model.output
            if isinstance(model_outputs, list):
                print(f"⚠️  Model has {len(model_outputs)} outputs, using last one")
                model_outputs = model_outputs[-1]
            
            # Create grad model
            grad_model = keras.models.Model(
                inputs=model.input,
                outputs=[last_conv_layer.output, model_outputs]
            )
            
            print(f"✅ Grad model created successfully")
            print(f"   Grad model inputs: {grad_model.input_shape}")
            print(f"   Grad model outputs: {[o.shape for o in grad_model.outputs]}")
            
            # Test gradient computation
            print(f"\n🔬 Testing gradient computation...")
            with tf.GradientTape() as tape:
                conv_output, preds = grad_model(test_input)
                
                print(f"   Conv output shape: {conv_output.shape}")
                print(f"   Predictions type: {type(preds)}")
                
                if isinstance(preds, list):
                    print(f"   ⚠️  Preds is a list with {len(preds)} elements")
                    preds = preds[-1] if len(preds) > 0 else preds
                
                print(f"   Predictions shape: {preds.shape}")
                pred_class = tf.argmax(preds[0])
                print(f"   Predicted class: {pred_class.numpy()}")
                
                class_channel = preds[:, pred_class]
                print(f"   Class channel shape: {class_channel.shape}")
            
            grads = tape.gradient(class_channel, conv_output)
            
            if grads is None:
                print(f"   ❌ Gradients are None - Cannot compute Grad-CAM")
                print(f"   This means the model might be frozen or gradients disabled")
            else:
                print(f"   ✅ Gradients computed successfully")
                print(f"   Gradient shape: {grads.shape}")
                print(f"   Gradient range: {grads.numpy().min():.6f} to {grads.numpy().max():.6f}")
                
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                conv_output_squeezed = conv_output[0]
                heatmap = conv_output_squeezed @ pooled_grads[..., tf.newaxis]
                heatmap = tf.squeeze(heatmap)
                heatmap = tf.maximum(heatmap, 0)
                
                if tf.reduce_max(heatmap) > 0:
                    heatmap = heatmap / tf.reduce_max(heatmap)
                
                heatmap_np = heatmap.numpy()
                print(f"   ✅ Heatmap generated")
                print(f"   Heatmap shape: {heatmap_np.shape}")
                print(f"   Heatmap range: {heatmap_np.min():.6f} to {heatmap_np.max():.6f}")
                print(f"   Heatmap mean: {heatmap_np.mean():.6f}")
                
                if heatmap_np.max() == heatmap_np.min():
                    print(f"   ⚠️  WARNING: Heatmap has no variation (all same value)")
                    print(f"   This will result in a uniform color overlay")
                else:
                    print(f"   ✅ Heatmap has good variation")
        
        except Exception as e:
            print(f"❌ Grad-CAM compatibility test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary and recommendations
    print(f"\n" + "="*70)
    print("📋 SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    if not conv_layers:
        print("❌ CRITICAL: No convolutional layers found")
        print("   → Grad-CAM cannot work with this model")
        print("   → Consider using a different visualization method")
    else:
        print(f"✅ Model has {len(conv_layers)} convolutional layers")
        print(f"✅ Last conv layer: {conv_layers[-1][1]}")
        
        if isinstance(model.output, list):
            print(f"⚠️  Model has multiple outputs ({len(model.output)})")
            print(f"   → app.py will use the LAST output")
            print(f"   → Make sure this is the correct classification output")
        
        print(f"\n📝 For app.py configuration:")
        print(f"   Last conv layer name: '{conv_layers[-1][1]}'")
        print(f"   If auto-detection fails, manually set this in app.py")
        
    print("\n" + "="*70)

except FileNotFoundError:
    print(f"\n❌ Model file not found: {MODEL_PATH}")
    print("Please check the path and try again.")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
