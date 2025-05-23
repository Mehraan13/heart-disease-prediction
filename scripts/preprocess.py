import os
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def main():
    # Create models directory
    os.makedirs("../models", exist_ok=True)

    # Load raw data
    df = pd.read_csv('../data/heart.csv')

    # 1) Handle missing values
    df['ca'].fillna(df['ca'].median(), inplace=True)
    df['thal'].fillna(df['thal'].mode()[0], inplace=True)

    # 2) Binary encode target (0 = no disease, 1 = disease)
    df['target'] = (df['target'] > 0).astype(int)

    # 3) Scale numeric features
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # 4) One-hot encode categorical features
    cat_cols = ['cp', 'restecg', 'slope', 'thal']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 5) Train-test split
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6) Persist split data and scaler
    split_path = os.path.join("../models", "split_data.pkl")
    scaler_path = os.path.join("../models", "scaler.pkl")
    joblib.dump((X_train, X_test, y_train, y_test), split_path)
    print(f"Saved train-test split to {split_path}")
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")


if __name__ == "__main__":
    main()
