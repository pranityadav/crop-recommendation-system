import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

from yield_preprocess import load_data, split_and_preprocess

MODEL_PATH = "models/yield_model.pkl"


def train():
    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor = split_and_preprocess(df)

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    # Evaluation on test set
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    joblib.dump(pipeline, MODEL_PATH)

    print("=== Yield Model Evaluation ===")
    print(f"MAE: {mae:.2f}")
    print(f"R2 : {r2:.3f}")
    print("Yield model saved to:", MODEL_PATH)


if __name__ == "__main__":
    train()

