# Brain Tumor Detection Web App

A complete web application for brain tumor detection using CNN with Grad-CAM visualization.

## Features

- 🧠 Brain tumor detection using a pre-trained VGG16 CNN model
- 🏥 **Specific tumor type classification**: Glioma, Meningioma, Pituitary
- 🔍 Grad-CAM (Gradient-weighted Class Activation Mapping) visualization
- 📊 Complete probability distribution for all tumor types
- 🌐 User-friendly web interface with color-coded results
- 🎨 Beautiful, responsive design
- ⚡ Real-time prediction with confidence scores

## Project Structure

```
.
├── app.py              # Flask backend server
├── index.html          # Frontend interface
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## ⚠️ Important: Python Version

**TensorFlow requires Python 3.9-3.12** (NOT 3.13)

If you have Python 3.13, see `SETUP_GUIDE.md` for instructions on installing a compatible version.

## Quick Start

### Option 1: Quick Start Script (Easiest)

```bash
# Make the script executable
chmod +x start.sh

# Run it
./start.sh
```

This script will:
- Check your Python version
- Test your model configuration
- Start the Flask server automatically

### Option 2: Manual Setup

## Setup Instructions

### 1. Test Your Model First

**IMPORTANT**: Before anything else, test your model configuration:

```bash
python test_model.py
```

This will tell you:
- How many classes your model has
- What tumor types it can detect
- If Grad-CAM will work
- What to configure in `app.py`

### 2. Install Dependencies

**Using Conda (Recommended for Python 3.13 users):**

```bash
conda create -n brain-tumor python=3.11 -y
conda activate brain-tumor
pip install flask flask-cors tensorflow numpy opencv-python Pillow
```

**Using Virtual Environment:**

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

### 3. Verify Model Path

Make sure your trained model is located at:
```
/Users/girijeshs/Downloads/Brave/VGG16_final.keras
```

If your model is in a different location, update the `MODEL_PATH` variable in `app.py`.

### 4. Configure Tumor Classes (if needed)

Open `app.py` and verify the `CLASS_LABELS` match your model's training.

See `MODEL_CONFIG.md` for detailed instructions.

### 5. Run the Flask Server

```bash
python app.py
```

The server will start at `http://localhost:5000`

You should see:
```
Model loaded successfully
Model input shape: (None, 224, 224, 3)
Model output shape: (None, 4)
```

### 6. Open the Frontend

Open `index.html` in your web browser, or serve it using a simple HTTP server:

```bash
# Option 1: Open directly
open index.html

# Option 2: Use Python's HTTP server
python3 -m http.server 8000
# Then visit http://localhost:8000
```

## Usage

1. Click the "Choose MRI Image" button to select an MRI scan
2. Click "Analyze Image" to get the prediction
3. View the results:
   - **Prediction**: Overall result (Tumor type or No Tumor)
   - **Tumor Type**: Specific classification (Glioma, Meningioma, or Pituitary)
   - **Confidence Score**: Model's confidence in the prediction
   - **All Probabilities**: Breakdown of confidence for each class
   - **Grad-CAM Heatmap**: Visual explanation showing which regions influenced the prediction

### Example Results

#### Tumor Detected:
```
Result: Glioma Tumor
🔴 Tumor Type: Glioma Tumor
Confidence: 87.43%

All Class Probabilities:
- Glioma Tumor:      87.43%
- Meningioma Tumor:   8.21%
- Pituitary Tumor:    3.12%
- No Tumor:           1.24%

[Grad-CAM visualization shown]
```

#### No Tumor:
```
Result: No Tumor
Confidence: 94.67%

All Class Probabilities:
- No Tumor:          94.67%
- Glioma Tumor:       2.31%
- Meningioma Tumor:   1.87%
- Pituitary Tumor:    1.15%

[Grad-CAM visualization shown]
```

## How Grad-CAM Works

Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes which parts of the image were most important for the model's decision. The heatmap overlay shows:

- **Red/Yellow regions**: Areas the model focused on most
- **Blue/Purple regions**: Areas with less influence on the prediction

This helps understand and interpret the model's decision-making process.

## API Endpoints

### `POST /predict`
Upload an image for prediction and Grad-CAM generation.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `file` (image file)

**Response:**
```json
{
    "prediction": "Glioma Tumor",
    "tumor_type": "Glioma Tumor",
    "is_tumor": true,
    "confidence": "87.43%",
    "predicted_class": 0,
    "raw_confidence": 0.8743,
    "all_probabilities": {
        "Glioma Tumor": 0.8743,
        "Meningioma Tumor": 0.0821,
        "Pituitary Tumor": 0.0312,
        "No Tumor": 0.0124
    },
    "gradcam_image": "base64_encoded_image_string",
    "gradcam_available": true
}
```

### `GET /model-info`
Get information about the loaded model.

**Response:**
```json
{
    "input_shape": "(None, 224, 224, 3)",
    "output_shape": "(None, 2)",
    "total_layers": 23,
    "conv_layers": ["block1_conv1", "block1_conv2", ...],
    "last_conv_layer": "block5_conv3"
}
```

## Customization

### Adjusting Class Labels

If your model has different classes, update the `CLASS_LABELS` dictionary in `app.py`:

```python
CLASS_LABELS = {0: "No Tumor", 1: "Tumor Detected"}
```

### Changing Image Preprocessing

Modify the `preprocess_image()` function in `app.py` if your model requires different preprocessing (e.g., different normalization, input size).

### Adjusting Grad-CAM Overlay

You can modify the overlay transparency by changing the `alpha` parameter in `create_gradcam_overlay()`:

```python
gradcam_image = create_gradcam_overlay(original_image, heatmap, alpha=0.4)
```

Lower alpha = more original image visible
Higher alpha = more heatmap visible

## Troubleshooting

### Python Version Issues
**Error**: `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'`
- **Cause**: Python 3.13 is not compatible with TensorFlow
- **Solution**: Use Python 3.9-3.12. See `SETUP_GUIDE.md` for installation instructions

### Model Configuration Issues
**Run this first**:
```bash
python test_model.py
```

This will diagnose:
- Model loading issues
- Class configuration problems
- Grad-CAM compatibility

### Grad-CAM Not Showing
1. Check Flask terminal for "Grad-CAM generated successfully"
2. Check browser console (F12) for errors
3. Run `python test_model.py` to find your last conv layer
4. Manually set the layer in `app.py` (see `MODEL_CONFIG.md`)

### Wrong Tumor Classifications
- Your `CLASS_LABELS` in `app.py` must match your training data
- Run `python test_model.py` to see your model's output shape
- Update `CLASS_LABELS` accordingly (see `MODEL_CONFIG.md`)

### CORS Issues
If you encounter CORS errors, make sure `flask-cors` is installed and CORS is enabled in `app.py`.

### Model Loading Errors
- Verify the model path is correct in `app.py`
- Ensure the model file is a valid Keras model (.keras or .h5 format)
- Check that TensorFlow version is compatible with your model
- Try: `python test_model.py`

### Port Already in Use
If port 5000 is already in use, change the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```
And update the `API_URL` in `index.html` accordingly.

## 📚 Documentation Files

- **`README.md`** (this file) - Overview and quick start
- **`SETUP_GUIDE.md`** - Detailed Python environment setup
- **`MODEL_CONFIG.md`** - How to configure tumor classes
- **`CHANGES.md`** - Summary of recent improvements
- **`test_model.py`** - Model diagnostic script

## Technologies Used

- **Backend**: Flask, TensorFlow/Keras
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **ML Libraries**: TensorFlow, NumPy, OpenCV, Pillow
- **Visualization**: Grad-CAM, OpenCV

## License

This project is open source and available for educational purposes.
