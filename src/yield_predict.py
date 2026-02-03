import joblib
import pandas as pd

MODEL_PATH = "models/yield_model.pkl"


def predict_yield(crop, soil_climate_features):
    """
    crop: str (e.g. 'rice')
    soil_climate_features: dict with keys:
        Temperature, Rainfall, Humidity,
        Soil_pH, Soil_Nitrogen, Soil_Phosphorus, Soil_Potassium
    """
    model = joblib.load(MODEL_PATH)

    input_row = {
        "Crop": crop,
        **soil_climate_features
    }

    input_df = pd.DataFrame([input_row])
    prediction = model.predict(input_df)[0]

    return float(prediction)
