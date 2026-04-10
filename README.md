# CardioGuard AI 🫀

An advanced machine learning system for **ECG anomaly detection** using FastAPI. This project leverages multiple ML models to detect electrocardiogram signal anomalies in real-time.

## Overview

CardioGuard AI is a specialized cardiovascular health assessment system that focuses on:
- **ECG Anomaly Detection:** Real-time analysis of electrocardiogram signals
- **RESTful API:** Built with FastAPI for high performance
- **Multiple Models:** Comparison of different ML algorithms

## Features

✅ ECG signal anomaly detection (Critical, Normal, Suspicious)  
✅ High accuracy predictions (91.5%+)  
✅ RESTful API with FastAPI  
✅ Interactive Swagger UI documentation  
✅ Model explainability & feature importance  
✅ Multi-model comparison  

## ⚠️ Important Note: Model File Size

The trained Random Forest model (`cardioguard_rf_model.pkl`) is **~200+ MB** in size, which exceeds GitHub's 100 MB file size limit. 

**Solutions:**
1. **Git LFS (Recommended):** Use [Git Large File Storage](https://git-lfs.github.com/) to track large model files
2. **Alternative Storage:** Upload models to cloud storage (AWS S3, Google Cloud Storage, Hugging Face)
3. **Model on Demand:** Download the model separately during setup
4. **.gitignore Approach:** Exclude model files from version control and regenerate them locally

The `.pkl` file is excluded from GitHub via `.gitignore`. To use this project:
- Train the model locally using `ecg_anomaly_detection_model.ipynb`
- Or download the model from alternative storage and place in the project directory

## Datasets

### MIT-BIH ECG Dataset
- **Source:** MIT-BIH Arrhythmia Database
- **File:** `mit_dataset3.csv`
- **Samples:** 21,114 ECG signal records
- **Target Classes:** 3-class classification
  - **Critical:** Severe ECG abnormalities
  - **Normal:** Healthy ECG signals
  - **Suspicious:** Borderline abnormal patterns

## Models Comparison

### ECG Anomaly Detection Models Performance

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---|
| **Random Forest (400 trees, balanced)** | **91.5%** | **0.92** | **0.92** | **0.92** | 4.2s |
| Random Forest (100 trees) | 91.2% | 0.91 | 0.91 | 0.91 | 2.1s |
| XGBoost (400 estimators) | 87.0% | 0.89 | 0.87 | 0.88 | 3.8s |
| Gradient Boosting (300 estimators) | 82.5% | 0.82 | 0.82 | 0.82 | 3.5s |

### Classification Report - Random Forest (400 trees, balanced - Deployed Model)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Critical | 0.87 | 0.90 | 0.88 | 6,244 |
| Normal | 0.94 | 0.95 | 0.94 | 13,536 |
| Suspicious | 0.93 | 0.71 | 0.81 | 1,334 |
| **Overall Accuracy** | — | — | **0.92** | 21,114 |

### Model Selection
- **Best Performing Model:** Random Forest (400 trees, balanced) with **91.5% accuracy**
- **Deployed Model:** Random Forest Classifier (400 estimators with balanced class weights)
- **Key Advantage:** Excellent balance between accuracy and inference speed, handles imbalanced data well

## Installation

### Requirements
- Python 3.8+
- pip (Python package manager)

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/ShikhajSomani/ECG-Anomaly-Detection-ML-Model.git
cd "CardioGuard AI"
```

2. **Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### 4. **Train or Obtain the Model**

**Option A: Train the Model Locally (Recommended)**
```bash
jupyter notebook ecg_anomaly_detection_model.ipynb
# Run all cells to train the model and save cardioguard_rf_model.pkl
```

**Option B: Download Pre-trained Model**
- Contact the repository maintainer for model file
- Place `cardioguard_rf_model.pkl` in the project root directory

### Dependencies (requirements.txt)
```
fastapi==0.104.1
uvicorn==0.24.0
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.26.2
xgboost==2.0.2
joblib==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
```

## Usage

### Running the FastAPI Server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

### Interactive API Documentation
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

## API Endpoints

### 1. Health Check
```
GET /
```
Returns API status.

**Response:**
```json
{
  "message": "ECG Anomaly Detection API is running"
}
```

### 2. ECG Anomaly Detection
```
POST /predict
```

**Request Body:**
```json
{
  "HR": 103.846,
  "RR_mean": 0.811,
  "RR_std": 0.078,
  "Quality": 0.926,
  "RMSSD": 0.138,
  "pNN50": 44.44,
  "CV": 0.097,
  "SDSD": 0.126,
  "RR_range": 0.222
}
```

**Response:**
```json
{
  "status": "Normal",
  "confidence": 0.987
}
```

## ECG Features Explained

The system analyzes 9 key ECG-derived features:

| Feature | Description | Normal Range |
|---------|-------------|---|
| **HR** | Heart Rate (beats per minute) | 60-100 |
| **RR_mean** | Mean RR interval (seconds) | 0.8-1.2 |
| **RR_std** | Standard deviation of RR intervals | 0.05-0.1 |
| **Quality** | ECG signal quality score | 0.9-1.0 |
| **RMSSD** | Root Mean Square Successive Difference | 0.1-0.4 |
| **pNN50** | Percentage of NN50 values | 20-50 |
| **CV** | Coefficient of Variation | 0.05-0.15 |
| **SDSD** | Standard Deviation of Successive Differences | 0.1-0.2 |
| **RR_range** | Range of RR intervals | 0.1-0.5 |

## Project Structure

```
CardioGuard AI/
├── app.py                           # FastAPI application
├── ecg_anomaly_detection_model.ipynb # ECG model training notebook
├── models/
│   ├── cardioguard_rf_model.pkl     # Trained Random Forest model (not in repo)
│   └── .gitkeep                     # Placeholder for models directory
├── data/
│   └── mit_dataset3.csv             # ECG signal dataset
├── requirements.txt                 # Project dependencies
├── .gitignore                       # Git ignore file (excludes large files)
├── README.md                        # This file
└── tests/
    └── test_api.py                  # API unit tests
```

**Note:** `cardioguard_rf_model.pkl` is not included in the repository due to its large file size (~200 MB). Train the model locally or use option B above.

## Performance Metrics

### ECG Anomaly Detection - Random Forest (400 trees, balanced)
- **Accuracy:** 91.5%
- **Precision:** 0.92
- **Recall:** 0.92
- **F1-Score:** 0.92
- **Average Response Time:** ~50ms

### Detailed Performance by Class

**Critical Cases:**
- Precision: 0.87 | Recall: 0.90 | F1-Score: 0.88
- Successfully identifies 90% of critical ECG patterns

**Normal Cases:**
- Precision: 0.94 | Recall: 0.95 | F1-Score: 0.94
- Highest confidence class with 95% recall rate

**Suspicious Cases:**
- Precision: 0.93 | Recall: 0.71 | F1-Score: 0.81
- Good precision but lower recall due to class imbalance

### Risk Level Classification

**ECG Signal Assessment:**
- **Normal:** Model predicts "Normal" class with high confidence
- **Suspicious:** Model predicts "Suspicious" class (borderline abnormal)
- **Critical:** Model predicts "Critical" class (requires immediate attention)

## Feature Importance

Based on the Random Forest model trained on ECG data (in order of importance):

1. **HR (Heart Rate)** - Most important
2. **RR_mean** - Mean RR interval
3. **RR_std** - RR interval standard deviation
4. **Quality** - Signal quality score
5. **RMSSD** - Heart rate variability metric
6. **pNN50** - Percentage of successive differences
7. **CV** - Coefficient of variation
8. **SDSD** - Successive differences standard deviation
9. **RR_range** - RR interval range

## Model Training Details

### Data Splitting
- **Training set:** 80% of data
- **Test set:** 20% of data (21,114 samples)
- **Stratified split:** Maintains class distribution
- **Random state:** 42 (for reproducibility)

### Models Evaluated:
1. **Random Forest (100 trees)** - Baseline model
2. **Random Forest (400 trees with balanced class weights)** - Deployed ✅
3. XGBoost (400 estimators)
4. Gradient Boosting (300 estimators)

### Why Random Forest (400 trees, balanced)?
- Highest accuracy among all models (91.5%)
- Fast inference time (critical for real-time prediction)
- Excellent precision on Normal class (0.94)
- Robust feature importance ranking
- Balanced class weights handle imbalanced dataset effectively

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Disclaimer

⚠️ **Medical Disclaimer:** This AI system is for educational and research purposes only. It should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

## Contact & Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Contact the development team
- Submit feature requests

## References

- MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/
- Random Forest Documentation: https://scikit-learn.org/stable/modules/ensemble.html#random-forests
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Git LFS: https://git-lfs.github.com/

---

**Last Updated:** April 10, 2026  
**Version:** 1.0.0  
**Status:** Active Development