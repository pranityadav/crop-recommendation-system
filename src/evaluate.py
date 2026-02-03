import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

from preprocess import load_data, split_data, scale_data

DATA_PATH = "data/Crop_recommendation.csv"
MODEL_PATH = "models/crop_model.pkl"
TARGET_COLUMN = "label"

def evaluate():
    df = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(df, TARGET_COLUMN)
    X_train_scaled, X_test_scaled, _ = scale_data(X_train, X_test)

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]

    # Predictions
    y_pred = model.predict(X_test_scaled)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # Cross-validation
    cv_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    cv_scores = cross_val_score(
        cv_model,
        X_train_scaled,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    print("\n=== Cross Validation Accuracy ===")
    print("Mean:", cv_scores.mean())
    print("Std :", cv_scores.std())

if __name__ == "__main__":
    evaluate()
