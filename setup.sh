#!/bin/bash
# Quick setup script for RAG Benchmark System

echo "================================"
echo "RAG Benchmark System - Quick Setup"
echo "================================"

# 1. Create .env file if not exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created. Please add your API keys!"
else
    echo "✓ .env file already exists"
fi

# 2. Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# 3. Activate virtual environment and install dependencies
echo "Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your API keys"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Run: python main.py prepare-data"
echo ""
