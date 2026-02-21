from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load("cardioguard_rf_model.pkl")

features = [
    "HR", "RR_mean", "RR_std", "Quality", "RMSSD", "pNN50", "CV", "SDSD", "RR_range"
]

@app.route("/")
def home():
    return "ECG API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    values = [data[f] for f in features]
    input_array = np.array(values).reshape(1,-1)

    prediction = model.predict(input_array)[0]

    return jsonify({"prediction": str(prediction)})