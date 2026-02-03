import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

DATA_PATH = "data/yield_data.csv"

FEATURES = [
    "Crop",
    "Temperature",
    "Rainfall",
    "Humidity",
    "Soil_pH",
    "Soil_Nitrogen",
    "Soil_Phosphorus",
    "Soil_Potassium"
]

TARGET = "Predicted_Yield"


def load_data():
    return pd.read_csv(DATA_PATH)


def split_and_preprocess(df):
    X = df[FEATURES]
    y = df[TARGET]

    cat_features = ["Crop"]
    num_features = [f for f in FEATURES if f != "Crop"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", "passthrough", num_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor = split_and_preprocess(df)

    print("Train size:", X_train.shape)
    print("Test size :", X_test.shape)
    print("Target OK :", y_train.notnull().all())
