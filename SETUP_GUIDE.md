# Setup Guide - Python Version Issue

## Problem
You're using Python 3.13, which is too new for TensorFlow and some other dependencies. TensorFlow currently supports Python 3.9-3.12.

## Solution Options

### Option 1: Use Conda (Recommended)

Conda makes it easy to create environments with specific Python versions.

```bash
# Install Miniconda if you don't have it
# Download from: https://docs.conda.io/en/latest/miniconda.html

# Create a new environment with Python 3.11
conda create -n brain-tumor python=3.11 -y

# Activate the environment
conda activate brain-tumor

# Install packages
pip install flask flask-cors tensorflow numpy opencv-python Pillow

# Run your app
python app.py
```

### Option 2: Use pyenv (Alternative)

```bash
# Install pyenv if you don't have it
brew install pyenv

# Install Python 3.11
pyenv install 3.11.7

# Set Python 3.11 for this directory
cd "/Users/girijeshs/Downloads/desktop,things/GitHub Repos/ann brain tumor"
pyenv local 3.11.7

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt

# Run your app
python app.py
```

### Option 3: Use System Python 3.11 or 3.12

If you have Python 3.11 or 3.12 installed:

```bash
# Find Python installations
ls /Library/Frameworks/Python.framework/Versions/

# Use specific Python version (adjust version as needed)
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

### Option 4: Install Python 3.11 via Homebrew

```bash
# Install Python 3.11
brew install python@3.11

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt

# Run your app
python app.py
```

## Quick Test

After setting up the correct Python version, test if TensorFlow works:

```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

## Recommended: Conda Approach

The easiest and most reliable method is using Conda:

```bash
# 1. Install Miniconda (if not installed)
#    Download from: https://docs.conda.io/en/latest/miniconda.html

# 2. Create environment
conda create -n brain-tumor python=3.11 -y

# 3. Activate
conda activate brain-tumor

# 4. Install all packages at once
conda install -c conda-forge flask flask-cors pillow opencv -y
pip install tensorflow

# 5. Run the app
python app.py
```

## After Setup

Once you have the correct Python environment:

1. **Start the Flask server:**
   ```bash
   python app.py
   ```
   Should see: `Running on http://127.0.0.1:5000`

2. **Open the frontend:**
   Open `index.html` in your browser

3. **Test with an MRI image:**
   Upload an image and click "Analyze Image"

## Troubleshooting

### Still getting errors?
- Check Python version: `python --version` (should be 3.9-3.12)
- Upgrade pip: `pip install --upgrade pip`
- Clear pip cache: `pip cache purge`

### Model path issues?
Verify your model location in `app.py`:
```python
MODEL_PATH = "/Users/girijeshs/Downloads/Brave/VGG16_final.keras"
```

### Port 5000 already in use?
Change the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```
And update `API_URL` in `index.html` to match.
