# TransCHD Web Demo (simplified)

## What this is
A simple Flask web app demonstrating a TransCHD-style segmentation pipeline with:
- Automatic dataset download attempt (Kaggle / direct link / Google Drive fallback).
- Minimal TransCHD-like model in TensorFlow/Keras.
- Frontend to upload an image and receive a segmentation overlay.

## Setup
1. Create a Python venv and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
