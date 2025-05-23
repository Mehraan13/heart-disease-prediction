import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def main():
    # Ensure models directory exists
    os.makedirs("../models", exist_ok=True)

    # Load split data
    split_path = os.path.join("../models", "split_data.pkl")
    X_train, X_test, y_train, y_test = joblib.load(split_path)

    # Instantiate Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        class_weight='balanced'
    )

    # Train model
    rf.fit(X_train, y_train)
    print("Random Forest trained.")

    # Evaluate AUC
    y_proba = rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"Random Forest ROC AUC: {auc:.3f}")

    # Save model
    model_path = os.path.join("../models", "random_forest.pkl")
    joblib.dump(rf, model_path)
    print(f"Saved Random Forest model to {model_path}")

if __name__ == "__main__":
    main()
