# app.py
import os
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
import tensorflow as tf

MODEL_PATH = "models/heart_classifier.h5"
SCALER_PATH = "models/scaler.pkl"

app = Flask(__name__, static_folder="static", template_folder="templates")

model = None
scaler = None

def load_model():
    global model, scaler
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("✅ Model and scaler loaded.")
    return model, scaler

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = np.array([list(data.values())], dtype=float)
        model, scaler = load_model()
        features = scaler.transform(features)
        pred = model.predict(features)
        prob_heart = float(pred[0][1])
        return jsonify({
            "probability_heart_disease": round(prob_heart, 3),
            "prediction": "Heart Disease" if prob_heart > 0.5 else "No Heart Disease"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=True)
