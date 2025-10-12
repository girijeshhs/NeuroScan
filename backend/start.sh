#!/bin/bash

# Quick Start Script for Brain Tumor Detection App
# This script helps you get started quickly

echo "🧠 Brain Tumor Detection App - Quick Start"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "❌ Python not found!"
    echo "Please install Python 3.9-3.12 first."
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "✅ Using: $PYTHON_CMD"
echo ""

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
echo "📌 Python version: $PYTHON_VERSION"

# Warn if Python 3.13
if [[ $PYTHON_VERSION == "3.13" ]]; then
    echo "⚠️  WARNING: Python 3.13 is not supported by TensorFlow!"
    echo "Please use Python 3.9, 3.10, 3.11, or 3.12"
    echo ""
    echo "See SETUP_GUIDE.md for instructions on installing a compatible Python version."
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "📋 Step 1: Testing your model configuration..."
echo "----------------------------------------------"
if [ -f "test_model.py" ]; then
    $PYTHON_CMD test_model.py
    echo ""
    read -p "Did the test pass? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Please fix the issues shown above before continuing."
        echo "See MODEL_CONFIG.md for help."
        exit 1
    fi
else
    echo "⚠️  test_model.py not found, skipping..."
fi

echo ""
echo "🚀 Step 2: Starting Flask server..."
echo "-----------------------------------"
echo "The server will start at: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start Flask app
$PYTHON_CMD app.py
