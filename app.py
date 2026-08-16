"""
Streamlit app for Assignment 2 - Machine Learning
Breast Cancer classification demo with 5 trained models.

Run locally:
    streamlit run app.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Breast Cancer Classifier Comparison", layout="wide")

# ---------------------------------------------------------------------------
# Load models, scaler, and metadata (cached so it only loads once)
# ---------------------------------------------------------------------------
MODEL_DIR = "model"

MODEL_FILES = {
    "Logistic Regression": "model_logistic_regression.joblib",
    "Decision Tree": "model_decision_tree.joblib",
    "kNN": "model_knn.joblib",
    "Naive Bayes": "model_naive_bayes.joblib",
    "Random Forest (Ensemble)": "model_random_forest_ensemble.joblib",
}


@st.cache_resource
def load_artifacts():
    models = {name: joblib.load(f"{MODEL_DIR}/{fname}") for name, fname in MODEL_FILES.items()}
    scaler = joblib.load(f"{MODEL_DIR}/scaler.joblib")
    with open(f"{MODEL_DIR}/model_meta.json") as f:
        meta = json.load(f)
    return models, scaler, meta


@st.cache_data
def load_precomputed_metrics():
    return pd.read_csv(f"{MODEL_DIR}/metrics_results.csv")


models, scaler, meta = load_artifacts()
feature_names = meta["feature_names"]
scaled_models = set(meta["scaled_models"])
class_names = meta["class_names"]

# ---------------------------------------------------------------------------
# Sidebar - navigation
# ---------------------------------------------------------------------------
st.title("🔬 Breast Cancer Classification - Model Comparison App")
st.caption(
    "ML Assignment 2 | Dataset: Breast Cancer Wisconsin (Diagnostic), UCI ML Repository "
    "| 569 instances, 30 features, binary classification"
)

st.sidebar.header("Controls")

page = st.sidebar.radio("Choose a view", ["Model Comparison (precomputed)", "Evaluate on Uploaded Test Data"])

# ---------------------------------------------------------------------------
# Page 1: Precomputed comparison table (from training run)
# ---------------------------------------------------------------------------
if page == "Model Comparison (precomputed)":
    st.subheader("📊 Evaluation Metrics - All Models (held-out test split)")
    metrics_df = load_precomputed_metrics()
    st.dataframe(metrics_df.set_index("ML Model Name"), use_container_width=True)

    st.subheader("Metric comparison chart")
    metric_to_plot = st.selectbox("Select metric to visualize", ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"])
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=metrics_df, x="ML Model Name", y=metric_to_plot, ax=ax, palette="viridis")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=20, ha="right")
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# Page 2: Upload test CSV, pick a model, see live metrics + confusion matrix
# ---------------------------------------------------------------------------
else:
    st.subheader("📁 Upload Test Data (CSV)")
    st.write(
        "Upload a CSV with the same 30 feature columns as the training data, "
        "plus a `target` column (0 = malignant, 1 = benign). "
        "A ready-made `test_data.csv` is included in this repo for this purpose."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    model_choice = st.selectbox("Select a model", list(models.keys()))

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "target" not in df.columns:
            st.error("The uploaded CSV must contain a 'target' column with the true labels.")
        else:
            missing_cols = [c for c in feature_names if c not in df.columns]
            if missing_cols:
                st.error(f"Uploaded CSV is missing required feature columns: {missing_cols[:5]}...")
            else:
                X = df[feature_names]
                y_true = df["target"]

                model = models[model_choice]
                X_input = scaler.transform(X) if model_choice in scaled_models else X

                y_pred = model.predict(X_input)
                y_proba = model.predict_proba(X_input)[:, 1]

                st.success(f"Predictions generated using **{model_choice}** on {len(df)} rows.")

                # --- Metrics ---
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
                col1.metric("AUC", f"{roc_auc_score(y_true, y_proba):.4f}")
                col2.metric("Precision", f"{precision_score(y_true, y_pred):.4f}")
                col2.metric("Recall", f"{recall_score(y_true, y_pred):.4f}")
                col3.metric("F1 Score", f"{f1_score(y_true, y_pred):.4f}")
                col3.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")

                # --- Confusion matrix ---
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(4, 3.5))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["malignant", "benign"],
                            yticklabels=["malignant", "benign"], ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

                # --- Classification report ---
                st.subheader("Classification Report")
                report = classification_report(y_true, y_pred, target_names=["malignant", "benign"], output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

                # --- Predictions preview ---
                st.subheader("Predictions Preview")
                preview = df.copy()
                preview["predicted"] = y_pred
                preview["predicted_proba_benign"] = y_proba
                st.dataframe(preview.head(20), use_container_width=True)
    else:
        st.info("Upload a CSV file to see predictions and evaluation metrics here.")

st.sidebar.markdown("---")
st.sidebar.caption("Assignment 2 - Machine Learning | M.Tech (AIML/DSE) | BITS Pilani WILP")
