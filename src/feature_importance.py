import joblib
import pandas as pd

MODEL_PATH = "models/crop_model.pkl"
DATA_PATH = "data/Crop_recommendation.csv"
TARGET_COLUMN = "label"

def show_feature_importance():
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]

    df = pd.read_csv(DATA_PATH)
    feature_names = df.drop(columns=[TARGET_COLUMN]).columns

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    print("\n=== Feature Importance ===")
    print(importance_df)

if __name__ == "__main__":
    show_feature_importance()
