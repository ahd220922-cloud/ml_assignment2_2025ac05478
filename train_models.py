"""
train_models.py
----------------
Trains 5 classification models on the Breast Cancer Wisconsin (Diagnostic)
dataset (UCI Machine Learning Repository / sklearn.datasets), evaluates them
on a held-out test split, saves the trained models + scaler for reuse in the
Streamlit app, and writes:
    - test_data.csv         -> held-out test set (features + true label) used
                                by the Streamlit app for demonstration
    - metrics_results.csv   -> comparison table of all 6 evaluation metrics
                                for each model (used to fill README.md)

Dataset details:
    Source: UCI ML Repository - Breast Cancer Wisconsin (Diagnostic) Data Set
            (bundled with scikit-learn via sklearn.datasets.load_breast_cancer)
    Instances: 569  (>= 500 required)
    Features:  30    (>= 12 required)
    Task: Binary classification (malignant = 0, benign = 1)
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target  # 0 = malignant, 1 = benign
feature_names = list(X.columns)

print(f"Dataset shape: {X.shape}, classes: {np.unique(y)}")

# ---------------------------------------------------------------------------
# 2. Train / test split (stratified)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------------
# 3. Save test data (features + true label) for use in the Streamlit app
# ---------------------------------------------------------------------------
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv("../test_data.csv", index=False)
print("Saved test_data.csv")

# ---------------------------------------------------------------------------
# 4. Scale features (fit on train only) - used for LR / kNN
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "scaler.joblib")

# ---------------------------------------------------------------------------
# 5. Define the 5 required models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": (LogisticRegression(max_iter=5000, random_state=RANDOM_STATE), True),
    "Decision Tree":       (DecisionTreeClassifier(random_state=RANDOM_STATE), False),
    "kNN":                 (KNeighborsClassifier(n_neighbors=7), True),
    "Naive Bayes":         (GaussianNB(), False),
    "Random Forest (Ensemble)": (RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE), False),
}

results = []

for name, (model, needs_scaling) in models.items():
    Xtr = X_train_scaled if needs_scaling else X_train
    Xte = X_test_scaled if needs_scaling else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]

    metrics = {
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)
    print(name, metrics)

    # Save the trained model
    fname = "model_" + name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"
    joblib.dump(model, fname)

# ---------------------------------------------------------------------------
# 6. Save comparison table
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("metrics_results.csv", index=False)
print("\nFinal comparison table:")
print(results_df.to_string(index=False))

# Save feature names + which models need scaling (used by the Streamlit app)
with open("model_meta.json", "w") as f:
    json.dump({
        "feature_names": feature_names,
        "scaled_models": [n for n, (m, s) in models.items() if s],
        "class_names": {"0": "malignant", "1": "benign"},
    }, f, indent=2)

print("\nDone. Models, scaler, metrics_results.csv, and model_meta.json saved in model/")
