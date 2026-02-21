from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

MODEL_PATH = "cardioguard_rf_model.pkl"

# -----------------------------
# 1️⃣ TRAIN MODEL FUNCTION
# -----------------------------
def train_model():
    print("Training model...")

    # Load dataset (must be in repo)
    df = pd.read_csv("mit_dataset.csv")

    X = df[[
        "HR", "RR_mean", "RR_std", "Quality",
        "RMSSD", "pNN50", "CV", "SDSD", "RR_range"
    ]]
    y = df["Label"]

    model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    return model


# -----------------------------
# 2️⃣ LOAD OR TRAIN MODEL
# -----------------------------
if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    model = joblib.load(MODEL_PATH)
else:
    model = train_model()
    joblib.dump(model, MODEL_PATH)
    print("Model trained and saved.")


FEATURES = [
    "HR", "RR_mean", "RR_std", "Quality",
    "RMSSD", "pNN50", "CV", "SDSD", "RR_range"
]


# -----------------------------
# 3️⃣ ROUTES
# -----------------------------
@app.route("/")
def home():
    return "ECG Anomaly Detection API is running"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    values = [data[f] for f in FEATURES]
    input_array = np.array(values).reshape(1, -1)

    prediction = model.predict(input_array)[0]

    return jsonify({"prediction": str(prediction)})