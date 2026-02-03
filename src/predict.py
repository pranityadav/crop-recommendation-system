import joblib
import numpy as np

MODEL_PATH = "models/crop_model.pkl"

def predict_crop(input_features, top_k=3):
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    scaler = saved["scaler"]

    input_scaled = scaler.transform([input_features])
    probabilities = model.predict_proba(input_scaled)[0]
    classes = model.classes_

    top_indices = np.argsort(probabilities)[::-1][:top_k]

    results = [
        (classes[i], float(probabilities[i]))
        for i in top_indices
    ]

    return results
