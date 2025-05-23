import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score
)

def main():
    # Ensure reports directory exists
    os.makedirs("../reports", exist_ok=True)

    # Load split data and model
    split_path = os.path.join("../models", "split_data.pkl")
    X_train, X_test, y_train, y_test = joblib.load(split_path)
    model_path = os.path.join("../models", "random_forest.pkl")
    rf = joblib.load(model_path)

    # Predict
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)

    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"Average Precision (AUC-PR): {avg_prec:.3f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp_cm = RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.savefig(os.path.join("../reports", "roc_curve.png"))
    plt.close()

    # Precision-Recall curve
    disp_pr = PrecisionRecallDisplay.from_predictions(y_test, y_proba)
    plt.savefig(os.path.join("../reports", "precision_recall_curve.png"))
    plt.close()

    # Confusion matrix heatmap
    plt.figure()
    cm = confusion_matrix(y_test, y_pred)
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join("../reports", "confusion_matrix.png"))
    plt.close()

    # Feature importances
    import pandas as pd
    feat_imp = pd.Series(rf.feature_importances_, index=[col for col in X_train.columns])
    feat_imp = feat_imp.sort_values(ascending=False)
    plt.figure(figsize=(8,6))
    feat_imp.head(10).plot(kind='bar')
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join("../reports", "feature_importances.png"))
    plt.close()

    print("Saved all evaluation plots to ../reports/")

if __name__ == "__main__":
    main()
