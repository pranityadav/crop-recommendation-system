import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def split_data(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

if __name__ == "__main__":
    df = load_data("data/Crop_recommendation.csv")

    # CHANGE THIS ONLY IF YOUR COLUMN NAME IS DIFFERENT
    TARGET_COLUMN = "label"

    X_train, X_test, y_train, y_test = split_data(df, TARGET_COLUMN)

    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    print("Train shape:", X_train_scaled.shape)
    print("Test shape:", X_test_scaled.shape)
    print("Target classes:", y_train.nunique())
