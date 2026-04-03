#!/bin/bash

# Ozone Prediction API - Setup Script
# Usage: bash setup.sh

set -e

echo "======================================="
echo "Ozone Prediction API - Setup"
echo "======================================="
echo ""

# Check Python
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.11+"
    exit 1
fi

PY_VERSION=$(python3 --version | awk '{print $2}')
echo "✓ Python $PY_VERSION found"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip setuptools wheel > /dev/null
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Setup environment
echo "Setting up environment..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✓ Created .env file (please edit with your settings)"
    echo "  Run: nano .env"
else
    echo "✓ .env file already exists"
fi
echo ""

# Create models directory
echo "Creating models directory..."
mkdir -p models
echo "✓ models/ directory ready (copy your trained models here)"
echo ""

# Ready to run
echo "======================================="
echo "✓ Setup complete!"
echo "======================================="
echo ""
echo "Next steps:"
echo "1. Edit .env with your Google API key"
echo "2. Copy trained models to models/"
echo "3. Run: python app.py"
echo "4. Visit: http://localhost:8000/docs"
echo ""
