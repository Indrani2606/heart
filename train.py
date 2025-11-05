# train.py
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.transchd import build_classifier
import download_data
import joblib

MODEL_OUT = "models/heart_classifier.h5"
SCALER_OUT = "models/scaler.pkl"

def main():
    path = download_data.main()
    csv_path = os.path.join(path, "heart.csv")
    df = pd.read_csv(csv_path)
    print("Dataset shape:", df.shape)

    # Features and labels
    X = df.drop("target", axis=1).values
    y = df["target"].values

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_OUT)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build model
    model = build_classifier(input_dim=X.shape[1], num_classes=2)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # Train
    model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.1)
    loss, acc = model.evaluate(X_test, y_test)
    print(f"✅ Test Accuracy: {acc:.3f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_OUT)
    print("Model saved at", MODEL_OUT)

if __name__ == "__main__":
    main()
