#!/bin/bash

# 🧠 Brain Tumor Detection - Quick Start Script
# This script starts both backend and frontend servers

echo "=================================="
echo "🧠 Brain Tumor Detection System"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo -e "${RED}❌ Backend directory not found!${NC}"
    echo "Please run this script from the project root directory."
    exit 1
fi

# Check if frontend directory exists
if [ ! -d "frontend" ]; then
    echo -e "${RED}❌ Frontend directory not found!${NC}"
    echo "Please run this script from the project root directory."
    exit 1
fi

# Start backend
echo -e "${BLUE}🚀 Starting Backend (Flask)...${NC}"
cd backend
python3 app.py &
BACKEND_PID=$!
echo -e "${GREEN}✅ Backend started (PID: $BACKEND_PID)${NC}"
echo "   📍 Running on: http://127.0.0.1:5000"
echo ""

# Wait for backend to initialize
sleep 3

# Start frontend
echo -e "${BLUE}🎨 Starting Frontend (React)...${NC}"
cd ../frontend
npm run dev &
FRONTEND_PID=$!
echo -e "${GREEN}✅ Frontend started (PID: $FRONTEND_PID)${NC}"
echo "   📍 Running on: http://localhost:5173"
echo ""

echo "=================================="
echo -e "${GREEN}✅ Application is running!${NC}"
echo "=================================="
echo ""
echo "Backend:  http://127.0.0.1:5000"
echo "Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Servers stopped"
    exit 0
}

# Trap Ctrl+C
trap cleanup INT

# Wait for processes
wait
