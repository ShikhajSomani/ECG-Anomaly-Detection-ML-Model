from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ECG Anomaly Detection API")

# ----------------------------
# Load trained model
# ----------------------------
try:
    model = joblib.load("cardioguard_rf_model.pkl")
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    model = None


# ----------------------------
# Define Input Schema
# ----------------------------
class ECGData(BaseModel):
    HR: float
    RR_mean: float
    RR_std: float
    Quality: float
    RMSSD: float
    pNN50: float
    CV: float
    SDSD: float
    RR_range: float


# ----------------------------
# Root Route
# ----------------------------
@app.get("/")
def root():
    return {"message": "ECG Anomaly Detection API is running"}


# ----------------------------
# Prediction Route
# ----------------------------
@app.post("/predict")
def predict(data: ECGData):

    if model is None:
        return {"error": "Model not loaded"}

    input_array = np.array([[
        data.HR,
        data.RR_mean,
        data.RR_std,
        data.Quality,
        data.RMSSD,
        data.pNN50,
        data.CV,
        data.SDSD,
        data.RR_range
    ]])

    prediction = model.predict(input_array)[0]
    confidence = model.predict_proba(input_array)[0].max()

    return {
        "status": prediction,
        "confidence": round(float(confidence), 3)
    }