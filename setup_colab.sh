#!/bin/bash
# TAPT Environment Setup Script for Google Colab
# Run this script after cloning your repository

echo "=== Setting up TAPT Environment for Google Colab ==="

# Navigate to project root
cd /content/tapt

# 1. Install Dassl.pytorch
echo "=== Installing Dassl.pytorch ==="
cd Dassl.pytorch
pip install -r requirements.txt
python setup.py develop
cd ..

# 2. Install CLIP from OpenAI
echo "=== Installing CLIP ==="
pip install git+https://github.com/openai/CLIP.git

# 3. Install additional dependencies
echo "=== Installing additional dependencies ==="
pip install ftfy regex tqdm scipy scikit-learn tabulate yacs gdown tb-nightly future

# 4. Set up environment variables
export DATA="/content/drive/MyDrive/datasets"
export PYTHONPATH="${PYTHONPATH}:/content/tapt/TAPT:/content/tapt/Dassl.pytorch"

echo "=== Environment setup complete ==="
echo "DATA path set to: $DATA"
echo "Make sure your datasets are in the correct location on Google Drive"
