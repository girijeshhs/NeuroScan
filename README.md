# 🧠 Brain Tumor Detection - AI-Powered MRI Analysis# Brain Tumor Detection Web App



[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)A complete web application for brain tumor detection using CNN with Grad-CAM visualization.

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

[![React](https://img.shields.io/badge/React-18-61dafb.svg)](https://reactjs.org/)## Features

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Advanced deep learning system for brain tumor detection and classification using Xception CNN architecture with explainable AI (Grad-CAM) visualization.- 🏥 **Specific tumor type classification**: Glioma, Meningioma, Pituitary

- 🔍 Grad-CAM (Gradient-weighted Class Activation Mapping) visualization

---- 📊 Complete probability distribution for all tumor types

- 🌐 User-friendly web interface with color-coded results

## 📁 Project Structure- 🎨 Beautiful, responsive design

- ⚡ Real-time prediction with confidence scores

```

ann-brain-tumor/## Project Structure

├── backend/              # Flask API + ML Model (Python)

├── frontend/             # React Web Application```

├── docs/                 # Documentation & Guides.

└── README.md            # You are here├── app.py              # Flask backend server

```├── index.html          # Frontend interface

├── requirements.txt    # Python dependencies

---└── README.md          # This file

```

## 🚀 Quick Start

## ⚠️ Important: Python Version

### Prerequisites

- Python 3.8+ **TensorFlow requires Python 3.9-3.12** (NOT 3.13)

- Node.js 16+

- pip & npmIf you have Python 3.13, see `SETUP_GUIDE.md` for instructions on installing a compatible version.



### 1. Start Backend (Flask API)## Quick Start

```bash

cd backend### Option 1: Quick Start Script (Easiest)

pip install -r requirements.txt

python3 app.py```bash

```# Make the script executable

Backend runs on: **http://127.0.0.1:5000**chmod +x start.sh



### 2. Start Frontend (React App)# Run it

```bash./start.sh

cd frontend```

npm install

npm run devThis script will:

```- Check your Python version

Frontend runs on: **http://localhost:5173**- Test your model configuration

- Start the Flask server automatically

### 3. Access Application

Open your browser and navigate to `http://localhost:5173`### Option 2: Manual Setup



---## Setup Instructions



## 📖 Detailed Setup### 1. Test Your Model First



### Backend Setup**IMPORTANT**: Before anything else, test your model configuration:



```bash```bash

cd backendpython test_model.py

```

# Install Python dependencies

pip install -r requirements.txtThis will tell you:

- How many classes your model has

# Configure model path in app.py (line 17)- What tumor types it can detect

MODEL_PATH = "/path/to/your/Xception_model.keras"- If Grad-CAM will work

- What to configure in `app.py`

# Start Flask server

python3 app.py### 2. Install Dependencies

```

**Using Conda (Recommended for Python 3.13 users):**

**Backend Endpoints:**

- `POST /predict` - Image upload & prediction```bash

- `GET /model-info` - Model architecture detailsconda create -n brain-tumor python=3.11 -y

- `GET /test-gradcam` - Test Grad-CAM functionalityconda activate brain-tumor

pip install flask flask-cors tensorflow numpy opencv-python Pillow

### Frontend Setup```



```bash**Using Virtual Environment:**

cd frontend

```bash

# Install Node dependenciespython3 -m venv venv

npm installsource venv/bin/activate  # On macOS/Linux

pip install -r requirements.txt

# Configure API endpoint in src/App.jsx (line 18)```

const API_URL = 'http://127.0.0.1:5000/predict'

### 3. Verify Model Path

# Start development server

npm run devMake sure your trained model is located at:

```

# Build for production/Users/girijeshs/Downloads/Brave/VGG16_final.keras

npm run build```

```

If your model is in a different location, update the `MODEL_PATH` variable in `app.py`.

---

### 4. Configure Tumor Classes (if needed)

## 🧪 Testing

Open `app.py` and verify the `CLASS_LABELS` match your model's training.

### Test Backend

```bashSee `MODEL_CONFIG.md` for detailed instructions.

cd backend

### 5. Run the Flask Server

# Test model configuration

python3 test_model.py```bash

python app.py

# Test preprocessing (determines correct method)```

python3 test_which_preprocessing.py path/to/mri_image.jpg

The server will start at `http://localhost:5000`

# Test Grad-CAM visualization

python3 test_gradcam_xception.py path/to/mri_image.jpgYou should see:

``````

Model loaded successfully

Model input shape: (None, 299, 299, 3)
Model output shape: (None, 4)

### Test Frontend

```bash

cd frontend```

npm test

npm run lint### 6. Open the Frontend

```

Open `index.html` in your web browser, or serve it using a simple HTTP server:

---

```bash

## 🎯 Features# Option 1: Open directly

open index.html

### Backend (Flask + TensorFlow)

- ✅ **Xception CNN** - 95%+ accuracy on brain tumor classification# Option 2: Use Python's HTTP server

- ✅ **Grad-CAM Visualization** - Explainable AI heatmapspython3 -m http.server 8000

- ✅ **4-Class Classification** - Glioma, Meningioma, Pituitary, No Tumor# Then visit http://localhost:8000

- ✅ **Xception Preprocessing** - Correct [-1, 1] normalization```

- ✅ **RESTful API** - Easy integration

- ✅ **CORS Enabled** - Frontend communication## Usage



### Frontend (React + Tailwind)1. Click the "Choose MRI Image" button to select an MRI scan

- ✅ **Modern Dark UI** - Professional medical interface2. Click "Analyze Image" to get the prediction

- ✅ **Drag & Drop Upload** - Easy image submission3. View the results:

- ✅ **Real-time Predictions** - Instant results   - **Prediction**: Overall result (Tumor type or No Tumor)

- ✅ **Grad-CAM Overlay** - Visual tumor localization   - **Tumor Type**: Specific classification (Glioma, Meningioma, or Pituitary)

- ✅ **Analysis History** - Track previous scans   - **Confidence Score**: Model's confidence in the prediction

- ✅ **Responsive Design** - Works on all devices   - **All Probabilities**: Breakdown of confidence for each class

- ✅ **Tumor Encyclopedia** - Detailed tumor information   - **Grad-CAM Heatmap**: Visual explanation showing which regions influenced the prediction



---### Example Results



## 🧠 Model Information#### Tumor Detected:

```

| Property | Value |Result: Glioma Tumor

|----------|-------|🔴 Tumor Type: Glioma Tumor

| Architecture | Xception (pretrained) |Confidence: 87.43%

| Input Size | 299×299×3 |

| Classes | 4 (Glioma, Meningioma, No Tumor, Pituitary) |All Class Probabilities:

| Preprocessing | `xception.preprocess_input()` → [-1, 1] |- Glioma Tumor:      87.43%

| Grad-CAM Layer | `block14_sepconv2_act` |- Meningioma Tumor:   8.21%

| Accuracy | 95%+ |- Pituitary Tumor:    3.12%

| Dataset | 7,000+ MRI scans |- No Tumor:           1.24%



---[Grad-CAM visualization shown]

```

## 📚 Documentation

#### No Tumor:

Comprehensive guides available in `/docs`:```

Result: No Tumor

| Document | Description |Confidence: 94.67%

|----------|-------------|

| `QUICK_FIX_GUIDE.md` | Quick troubleshooting reference |All Class Probabilities:

| `XCEPTION_FIX_APPLIED.md` | Latest Grad-CAM fixes explained |- No Tumor:          94.67%

| `SETUP_GUIDE.md` | Detailed setup instructions |- Glioma Tumor:       2.31%

| `FINAL_CONFIGURATION.md` | Working configuration settings |- Meningioma Tumor:   1.87%

| `GRADCAM_FIX_SUMMARY.md` | Summary of all Grad-CAM fixes |- Pituitary Tumor:    1.15%



---[Grad-CAM visualization shown]

```

## 🔧 Configuration

## How Grad-CAM Works

### Backend Configuration (`backend/app.py`)

Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes which parts of the image were most important for the model's decision. The heatmap overlay shows:

```python

# Model path- **Red/Yellow regions**: Areas the model focused on most

MODEL_PATH = "/Users/girijeshs/Desktop/chrome/Xception_95pct_model.keras"- **Blue/Purple regions**: Areas with less influence on the prediction



# Class labels (must match training order)This helps understand and interpret the model's decision-making process.

CLASS_LABELS = {

    0: "Glioma Tumor",## API Endpoints

    1: "Meningioma Tumor", 

    2: "No Tumor",### `POST /predict`

    3: "Pituitary Tumor"Upload an image for prediction and Grad-CAM generation.

}

**Request:**

# Server settings- Method: POST

app.run(debug=True, host='0.0.0.0', port=5000)- Content-Type: multipart/form-data

```- Body: `file` (image file)



### Frontend Configuration (`frontend/src/App.jsx`)**Response:**

```json

```javascript{

// API endpoint    "prediction": "Glioma Tumor",

const API_URL = 'http://127.0.0.1:5000/predict'    "tumor_type": "Glioma Tumor",

    "is_tumor": true,

// App settings    "confidence": "87.43%",

const [darkMode] = useState(true)  // Dark mode enabled    "predicted_class": 0,

```    "raw_confidence": 0.8743,

    "all_probabilities": {

---        "Glioma Tumor": 0.8743,

        "Meningioma Tumor": 0.0821,

## 🛠️ Tech Stack        "Pituitary Tumor": 0.0312,

        "No Tumor": 0.0124

### Backend    },

- **Framework**: Flask 2.x    "gradcam_image": "base64_encoded_image_string",

- **ML**: TensorFlow 2.x, Keras    "gradcam_available": true

- **Image Processing**: OpenCV, Pillow}

- **Scientific Computing**: NumPy```

- **CORS**: Flask-CORS

### `GET /model-info`

### FrontendGet information about the loaded model.

- **Framework**: React 18

- **Build Tool**: Vite**Response:**

- **Styling**: Tailwind CSS 3```json

- **HTTP Client**: Axios{

- **Icons**: Lucide React
- **Animations**: Framer Motion

---

**Response:**
```json
{
    "input_shape": "(None, 299, 299, 3)",
    "output_shape": "(None, 4)",
    "total_layers": 136,
    "conv_layers": ["block1_conv1", "block1_conv2", ...],
    "last_conv_layer": "block14_sepconv2_act"

## 📊 API Reference}

```

### POST /predict

Upload MRI image for classification and Grad-CAM visualization.## Customization



**Request:**### Adjusting Class Labels

```bash

curl -X POST \If your model has different classes, update the `CLASS_LABELS` dictionary in `app.py`:

  -F "file=@mri_scan.jpg" \

  http://127.0.0.1:5000/predict```python

```CLASS_LABELS = {0: "No Tumor", 1: "Tumor Detected"}

```

**Response:**

```json### Changing Image Preprocessing

{

  "prediction": "Meningioma Tumor",Modify the `preprocess_image()` function in `app.py` if your model requires different preprocessing (e.g., different normalization, input size).

  "confidence": "96.84%",

  "tumor_type": "Meningioma Tumor",### Adjusting Grad-CAM Overlay

  "is_tumor": true,

  "all_probabilities": {You can modify the overlay transparency by changing the `alpha` parameter in `create_gradcam_overlay()`:

    "Glioma Tumor": 0.0231,

    "Meningioma Tumor": 0.9684,```python

    "No Tumor": 0.0012,gradcam_image = create_gradcam_overlay(original_image, heatmap, alpha=0.4)

    "Pituitary Tumor": 0.0073```

  },

  "gradcam_image": "base64_encoded_image",Lower alpha = more original image visible

  "gradcam_available": trueHigher alpha = more heatmap visible

}

```## Troubleshooting



### GET /model-info### Python Version Issues

Get model architecture information.**Error**: `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'`

- **Cause**: Python 3.13 is not compatible with TensorFlow

### GET /test-gradcam- **Solution**: Use Python 3.9-3.12. See `SETUP_GUIDE.md` for installation instructions

Test Grad-CAM with random image.

### Model Configuration Issues

---**Run this first**:

```bash

## 🐛 Troubleshootingpython test_model.py

```

### Backend Issues

This will diagnose:

**Problem**: Model not loading- Model loading issues

```bash- Class configuration problems

# Check model path in backend/app.py- Grad-CAM compatibility

MODEL_PATH = "/correct/path/to/model.keras"

```### Grad-CAM Not Showing

1. Check Flask terminal for "Grad-CAM generated successfully"

**Problem**: Wrong preprocessing2. Check browser console (F12) for errors

```bash3. Run `python test_model.py` to find your last conv layer

cd backend4. Manually set the layer in `app.py` (see `MODEL_CONFIG.md`)

python3 test_which_preprocessing.py path/to/mri.jpg

# Use the method with higher confidence### Wrong Tumor Classifications

```- Your `CLASS_LABELS` in `app.py` must match your training data

- Run `python test_model.py` to see your model's output shape

**Problem**: Grad-CAM not working- Update `CLASS_LABELS` accordingly (see `MODEL_CONFIG.md`)

```bash

cd backend### CORS Issues

python3 test_gradcam_xception.py path/to/mri.jpgIf you encounter CORS errors, make sure `flask-cors` is installed and CORS is enabled in `app.py`.

# Check console output for errors

```### Model Loading Errors

- Verify the model path is correct in `app.py`

### Frontend Issues- Ensure the model file is a valid Keras model (.keras or .h5 format)

- Check that TensorFlow version is compatible with your model

**Problem**: Cannot connect to API- Try: `python test_model.py`

- Verify backend is running: `http://127.0.0.1:5000`

- Check CORS is enabled in `backend/app.py`### Port Already in Use

- Verify API_URL in `frontend/src/App.jsx`If port 5000 is already in use, change the port in `app.py`:

```python

**Problem**: Build failsapp.run(debug=True, host='0.0.0.0', port=5001)

```bash```

cd frontendAnd update the `API_URL` in `index.html` accordingly.

rm -rf node_modules package-lock.json

npm install## 📚 Documentation Files

npm run dev

```- **`README.md`** (this file) - Overview and quick start

- **`SETUP_GUIDE.md`** - Detailed Python environment setup

---- **`MODEL_CONFIG.md`** - How to configure tumor classes

- **`CHANGES.md`** - Summary of recent improvements

## 📈 Performance- **`test_model.py`** - Model diagnostic script



- **Prediction Time**: ~3-5 seconds per image## Technologies Used

- **Grad-CAM Generation**: ~2-3 seconds

- **Model Size**: ~88 MB- **Backend**: Flask, TensorFlow/Keras

- **Memory Usage**: ~500 MB (with TensorFlow)- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)

- **ML Libraries**: TensorFlow, NumPy, OpenCV, Pillow

---- **Visualization**: Grad-CAM, OpenCV



## 🔒 Security## License



- ✅ CORS configured for local developmentThis project is open source and available for educational purposes.

- ✅ File type validation (images only)
- ✅ Input sanitization
- ⚠️ **Production**: Add authentication, HTTPS, rate limiting

---

## 🚢 Deployment

### Backend (Production)
```bash
cd backend
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Frontend (Production)
```bash
cd frontend
npm run build
# Deploy dist/ folder to hosting service
```

### Docker (Optional)
```dockerfile
# Backend Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ .
CMD ["python", "app.py"]
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 👨‍💻 Development

### Backend Development
```bash
cd backend
python3 app.py
# Edit files, Flask auto-reloads
```

### Frontend Development
```bash
cd frontend
npm run dev
# Edit files, Vite hot-reloads
```

### Add New Features
1. Backend: Add endpoint in `backend/app.py`
2. Frontend: Create component in `frontend/src/components/`
3. Update documentation in `docs/`

---

## 🎓 Learn More

- [Xception Paper](https://arxiv.org/abs/1610.02357)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [React Documentation](https://react.dev/)

---

## ⚕️ Medical Disclaimer

**This tool is for research and educational purposes only.**

❗ This application:
- Should NOT be used as the sole basis for clinical diagnosis
- Is NOT a substitute for professional medical advice
- Requires validation by qualified medical professionals
- May produce false positives or false negatives

Always consult qualified healthcare providers for medical decisions.

---

## 📞 Support

Having issues? Check these resources:

1. **Quick Fix Guide**: `docs/QUICK_FIX_GUIDE.md`
2. **Run Diagnostics**: `python3 backend/test_model.py`
3. **Check Preprocessing**: `python3 backend/test_which_preprocessing.py`
4. **Test Grad-CAM**: `python3 backend/test_gradcam_xception.py`

---

## ✨ Acknowledgments

- Dataset: Public brain tumor MRI repositories
- Base Model: Xception by Google
- Grad-CAM: Visualization technique by Selvaraju et al.

---

**Built with ❤️ for medical AI research**

**Status**: ✅ Production Ready  
**Last Updated**: 2025-10-12  
**Version**: 2.0
