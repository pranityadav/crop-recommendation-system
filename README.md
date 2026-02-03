# 🌾 Crop & Yield Recommendation System

This project is a multi-stage machine learning system that:
1. Recommends the most suitable crops based on soil and climate conditions
2. Predicts the expected yield for each recommended crop

## System Design
The system uses **two separate ML models**:

### 1. Crop Recommendation (Classification)
- Input: Soil nutrients + climate features
- Output: Top crop recommendations with confidence
- Model: Random Forest Classifier

### 2. Yield Prediction (Regression)
- Input: Soil + climate + selected crop
- Output: Expected crop yield (numeric)
- Model: Random Forest Regressor

This separation avoids mixing objectives and improves interpretability.

## Input Features
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

## Model Evaluation
**Crop Model**
- Accuracy: ~99%
- Cross-validation used
- Confusion matrix & classification report analyzed

**Yield Model**
- Metric: MAE, R²
- Yield data is synthetic and used for demonstration purposes

## Tech Stack
- Python
- Pandas, NumPy
- scikit-learn
- Streamlit

## Disclaimer
Yield data is synthetic and the project is intended for educational purposes only.
