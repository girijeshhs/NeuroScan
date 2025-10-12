# Brain Tumor Detection - React Frontend

A modern, responsive React frontend built with Vite and Tailwind CSS for brain tumor detection using AI.

## 🚀 Features

- ✨ Modern UI with Tailwind CSS
- 🌓 Dark mode support
- 📱 Fully responsive design
- 🎭 Smooth animations with Framer Motion
- 📤 Drag-and-drop file upload
- 🎨 Grad-CAM visualization (only for tumor cases)
- 📊 Detailed probability breakdown
- ⚡ Fast development with Vite
- 🎯 Clean, medical-themed design

## 📋 Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- Flask backend running at `http://127.0.0.1:5000`

## 🛠️ Installation

1. **Navigate to the frontend directory:**
   ```bash
   cd brain-tumor-frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

## 🏃 Running the App

1. **Start the development server:**
   ```bash
   npm run dev
   ```

2. **Open your browser:**
   Navigate to `http://localhost:3000`

3. **Make sure your Flask backend is running:**
   ```bash
   # In your Flask backend directory
   python app.py
   ```

## 📦 Build for Production

```bash
npm run build
```

The build output will be in the `dist` directory.

To preview the production build:
```bash
npm run preview
```

## 🎨 Features Breakdown

### Upload Interface
- Drag-and-drop or click to upload MRI images
- Image preview before analysis
- Support for JPG, PNG, JPEG formats

### Analysis Results
- **Prediction Display**: Clear indication of tumor presence
- **Tumor Type**: Specific classification (Glioma, Meningioma, Pituitary)
- **Confidence Score**: Percentage confidence in prediction
- **Probability Breakdown**: Detailed probabilities for all classes
- **Grad-CAM Visualization**: Heat map showing tumor location (only when tumor detected)

### Dark Mode
- Toggle between light and dark themes
- Persistent across sessions
- Smooth transitions

### Responsive Design
- Mobile-friendly layout
- Adapts to all screen sizes
- Touch-optimized interactions

## 🔧 Configuration

### API Endpoint
The Flask backend URL is configured in `src/App.jsx`:
```javascript
const API_URL = 'http://127.0.0.1:5000/predict'
```

### Tailwind Theme
Customize colors and styling in `tailwind.config.js`:
```javascript
theme: {
  extend: {
    colors: {
      medical: {
        blue: '#4A90E2',
        darkblue: '#2C5F8D',
        light: '#E8F4F8',
      },
    },
  },
}
```

## 📁 Project Structure

```
brain-tumor-frontend/
├── src/
│   ├── components/
│   │   ├── UploadBox.jsx       # File upload component
│   │   ├── ResultCard.jsx      # Results display
│   │   ├── LoadingSpinner.jsx  # Loading animation
│   │   └── ErrorToast.jsx      # Error notifications
│   ├── App.jsx                 # Main application
│   ├── main.jsx               # Entry point
│   └── index.css              # Global styles
├── public/                     # Static assets
├── index.html                 # HTML template
├── vite.config.js            # Vite configuration
├── tailwind.config.js        # Tailwind configuration
├── postcss.config.js         # PostCSS configuration
└── package.json              # Dependencies
```

## 🎯 Backend API Requirements

Your Flask backend should return a JSON response in this format:

```json
{
  "prediction": "Glioma Tumor",
  "tumor_type": "Glioma Tumor",
  "is_tumor": true,
  "confidence": "87.43%",
  "raw_confidence": 0.8743,
  "predicted_class": 0,
  "all_probabilities": {
    "Glioma Tumor": 0.8743,
    "Meningioma Tumor": 0.0821,
    "Pituitary Tumor": 0.0312,
    "No Tumor": 0.0124
  },
  "gradcam_image": "base64_encoded_string",
  "gradcam_available": true
}
```

## 🐛 Troubleshooting

### CORS Issues
If you get CORS errors, make sure your Flask backend has CORS enabled:
```python
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
```

### Connection Refused
- Ensure Flask backend is running at `http://127.0.0.1:5000`
- Check that port 5000 is not blocked by firewall
- Verify the API_URL in `src/App.jsx`

### Images Not Displaying
- Check that base64 image data is properly formatted
- Verify the Flask backend is returning `gradcam_image` field
- Check browser console for errors

## 🌟 Key Technologies

- **React 18** - UI library
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **Framer Motion** - Animation library
- **Axios** - HTTP client
- **Lucide React** - Icon library

## 👨‍💻 Developer

**Developed by Girijesh S**

## 📄 License

This project is for educational purposes only.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

---

**Happy Coding! 🚀**
