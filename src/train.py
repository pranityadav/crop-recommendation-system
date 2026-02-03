import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from preprocess import load_data, split_data, scale_data

DATA_PATH = "data/Crop_recommendation.csv"
MODEL_PATH = "models/crop_model.pkl"
TARGET_COLUMN = "label"

def train():
    df = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(df, TARGET_COLUMN)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    joblib.dump(
        {
            "model": model,
            "scaler": scaler
        },
        MODEL_PATH
    )

    print(f"Baseline Accuracy: {acc:.4f}")
    print("Model saved to:", MODEL_PATH)

if __name__ == "__main__":
    train()
