from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("cardioguard_rf_model.pkl","rb")

FEATURES = [
    "HR", "RR_mean", "RR_std", "Quality",
    "RMSSD", "pNN50", "CV", "SDSD", "RR_range"
]

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