# 🫀 ECG Anomaly Detection using Machine Learning

A Machine Learning API that classifies ECG heart rhythm into:

- ✅ Normal  
- ⚠ Suspicious  
- 🚨 Critical  

Built using **Random Forest + Advanced HRV Feature Engineering** and deployed using **FastAPI on Render**.

---

## 📌 Problem Statement

Early detection of abnormal heart rhythms is critical for preventing serious cardiac events.

This project processes ECG-derived RR interval data and classifies heart rhythm into:

- **Normal**
- **Suspicious**
- **Critical**

---

## 📊 Dataset

- MIT-BIH Arrhythmia Dataset
- 100,000+ processed samples
- Sliding window applied on RR intervals
- Converted from time-series ECG signal to structured ML dataset

---

## 🧠 Feature Engineering

### Initial Features (4)

- HR (Heart Rate)
- RR_mean
- RR_std
- Quality

Initial Model Performance:

- Accuracy: ~85%
- Suspicious Recall: 0.54

---

### ➕ Advanced HRV Features Added

- RMSSD
- pNN50
- CV (Coefficient of Variation)
- SDSD
- RR_range

These features capture short-term beat-to-beat variability and rhythm instability.

---

## 📈 Model Improvement

| Version | Features | Accuracy | Suspicious Recall |
|----------|----------|-----------|------------------|
| Initial Model | 4 | ~85% | 0.54 |
| Improved Model | 9 | **91%+** | **0.70+** |

Feature engineering significantly improved minority class detection.

---

## 🏗 Final Model Configuration

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
```

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- NumPy
- Pandas
- FastAPI
- Uvicorn
- Render (Cloud Deployment)

---

## Run Locally

Install dependencies
```python
pip install -r requirements.txt
```

Run the server
```python
uvicorn app:app --reload
```

Open
```python
http://127.0.0.1:8000/docs
```
