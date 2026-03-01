import sys
from pathlib import Path

# Ensure Python always finds model.py and preprocess.py in the same src/ folder,
# even when the app is launched from outside the src/ directory.
SRC_DIR  = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix

# Import our custom preprocessing and evaluation functions
from preprocess import preprocess_uploaded_data
from model import evaluate_dataset

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION — must be the very first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Patient Risk Assessment System",
    page_icon="🏥",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — polishes the look of the app
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .main-title  { font-size: 2.3rem; font-weight: 700; color: #1f3d6b; }
        .subtitle    { font-size: 1.05rem; color: #555; margin-bottom: 1rem; }
        .metric-card { background: #f0f4ff; border-radius: 10px; padding: 12px; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🏥 Intelligent Patient Risk Assessment System</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Upload patient data or use a sample dataset to assess diabetes risk.</p>',
    unsafe_allow_html=True,
)
st.warning(
    "⚠️ **Disclaimer:** This tool is for educational purposes only and is not a "
    "substitute for professional medical advice. Always consult a qualified healthcare provider."
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — SIDEBAR: DATA SOURCE SELECTION
# The user can upload their own CSV/XLSX or pick one of the built-in samples.
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("� Data Source")

    data_source = st.radio(
        "Choose how to load patient data:",
        options=[
            "📤 Upload my own dataset (CSV or XLSX)",
            "🔴 Use Sample: High Risk Patients",
            "🟢 Use Sample: Low Risk Patients",
        ],
    )

    raw_df = None  # will hold the loaded DataFrame

    if data_source == "📤 Upload my own dataset (CSV or XLSX)":
        uploaded_file = st.file_uploader(
            "Upload a CSV or Excel file",
            type=["csv", "xlsx"],
            help="File must contain: Pregnancies, Glucose, BloodPressure, "
                 "SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age",
        )
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".xlsx"):
                raw_df = pd.read_excel(uploaded_file, engine="openpyxl")
            else:
                raw_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded **{uploaded_file.name}** — {len(raw_df)} rows")

    elif data_source == "🔴 Use Sample: High Risk Patients":
        raw_df = pd.read_csv(BASE_DIR / "data" / "sample_high_risk.csv")
        st.success(f"✅ Loaded sample_high_risk.csv — {len(raw_df)} rows")

    else:  # Low Risk sample
        raw_df = pd.read_csv(BASE_DIR / "data" / "sample_low_risk.csv")
        st.success(f"✅ Loaded sample_low_risk.csv — {len(raw_df)} rows")

# If no data has been loaded yet, prompt the user and halt execution
if raw_df is None:
    st.info("👈 Please select a data source from the sidebar to begin.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — PREPROCESS & PREDICT
# Validate columns, clean data, scale features, then run batch inference.
# ─────────────────────────────────────────────────────────────────────────────
try:
    X_scaled, y_true = preprocess_uploaded_data(raw_df)
except ValueError as ve:
    # Raised when required columns are missing
    st.error(f"❌ **Column validation failed:** {ve}")
    st.stop()
except Exception as e:
    st.error(f"❌ An unexpected error occurred: {e}")
    st.stop()

# Run batch predictions and compute metrics (if Outcome column is present)
results = evaluate_dataset(X_scaled, y_true)

# ── Build the output DataFrame ─────────────────────────────────────────────
output_df = raw_df.copy().reset_index(drop=True)

# Map numeric predictions to readable labels
output_df["Prediction"] = [
    "🔴 Diabetes Risk" if p == 1 else "✅ No Diabetes"
    for p in results["predictions"]
]

# Probability as a percentage rounded to 1 decimal place
probs = results["probabilities"]
output_df["Risk Probability %"] = (probs * 100).round(1)

# Confidence = how certain the model is, regardless of which class it picked
output_df["Confidence"] = [
    f"{(p if p >= 0.5 else 1 - p) * 100:.1f}%"
    for p in probs
]

# Risk tier based on probability threshold
def risk_level(p):
    if p >= 0.7:
        return "High"
    elif p >= 0.4:
        return "Medium"
    else:
        return "Low"

output_df["Risk Level"] = [risk_level(p) for p in probs]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — THREE-TAB LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📋 Predictions & Risk Assessment",
    "📊 Model Evaluation Metrics",
    "📈 Trend Insights",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Predictions & Risk Assessment
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Patient-Level Risk Predictions")

    # ── Summary metrics row ──────────────────────────────────────────────────
    total      = len(output_df)
    high_count = (output_df["Risk Level"] == "High").sum()
    med_count  = (output_df["Risk Level"] == "Medium").sum()
    low_count  = (output_df["Risk Level"] == "Low").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Total Patients",  total)
    c2.metric("🔴 High Risk",       high_count)
    c3.metric("🟡 Medium Risk",     med_count)
    c4.metric("🟢 Low Risk",        low_count)

    st.markdown("---")

    # ── Raw input data expander ───────────────────────────────────────────────
    with st.expander("📄 View Raw Input Data", expanded=False):
        st.dataframe(raw_df, use_container_width=True)

    # ── Full results table ────────────────────────────────────────────────────
    st.markdown("#### 🗂️ Full Prediction Results")
    display_cols = [
        "Pregnancies", "Glucose", "BMI", "Age",
        "Prediction", "Risk Probability %", "Confidence", "Risk Level",
    ]
    st.dataframe(output_df[display_cols], use_container_width=True)

    # ── Download button ───────────────────────────────────────────────────────
    csv_bytes = output_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv_bytes,
        file_name="risk_assessment_results.csv",
        mime="text/csv",
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Evaluation Metrics
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Performance Evaluation")

    if results["metrics_available"]:
        # ── Four metric cards ─────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("✅ Accuracy",   f"{results['accuracy'] * 100:.2f}%")
        m2.metric("📉 RMSE",       f"{results['rmse']:.4f}")
        m3.metric("📐 MAE",        f"{results['mae']:.4f}")
        m4.metric("📈 R² Score",   f"{results['r2']:.4f}")

        st.markdown("---")

        # ── Confusion matrix heatmap ──────────────────────────────────────────
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_true, results["predictions"])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Diabetes", "Diabetes"],
            yticklabels=["No Diabetes", "Diabetes"],
            linewidths=0.5,
            ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── Metric explanations ───────────────────────────────────────────────
        with st.expander("ℹ️ What do these metrics mean?", expanded=False):
            st.markdown(
                """
**✅ Accuracy**
The percentage of patients whose diabetes status was correctly predicted.
For example, 75% accuracy means the model got 3 out of every 4 patients right.
However, accuracy alone can be misleading if the data is imbalanced — that's why we also look at RMSE, MAE, and R².

**📉 RMSE (Root Mean Square Error)**
RMSE measures how far off the model's predictions are from the true labels, on average — but it penalises large errors more than small ones.
A lower RMSE indicates better, more consistent predictions.
Since our labels are binary (0 or 1), an RMSE close to 0 is ideal.

**📐 MAE (Mean Absolute Error)**
MAE is the average absolute difference between predicted and actual values.
It is easier to interpret than RMSE because it is in the same units as the target.
For this binary task, an MAE of 0.25 means the model is off by 0.25 on average per prediction.

**📈 R² Score (Coefficient of Determination)**
R² tells us how much of the variation in the Outcome the model can explain.
A score of 1.0 means perfect prediction; a score of 0.0 means the model is no better than guessing the mean.
Negative values indicate that a simple average would outperform the model.
                """
            )

    else:
        # No Outcome column was found in the uploaded file
        st.info(
            "ℹ️ Your uploaded dataset does not have an **'Outcome'** column, so "
            "evaluation metrics cannot be calculated. Predictions are still shown in Tab 1."
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Trend Insights
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Visual Trend Insights")

    # ── Row 1: Prediction distribution | Risk Level distribution ─────────────
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("##### Prediction Distribution")
        fig, ax = plt.subplots()
        pred_counts = output_df["Prediction"].value_counts()
        ax.bar(pred_counts.index, pred_counts.values, color=["#2ecc71", "#e74c3c"])
        ax.set_title("Prediction Distribution")
        ax.set_ylabel("Count")
        ax.set_xlabel("Prediction")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with r1c2:
        st.markdown("##### Risk Level Distribution")
        fig, ax = plt.subplots()
        risk_counts = output_df["Risk Level"].value_counts()
        colors = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"}
        bar_colors = [colors.get(lvl, "#3498db") for lvl in risk_counts.index]
        ax.bar(risk_counts.index, risk_counts.values, color=bar_colors)
        ax.set_title("Risk Level Distribution")
        ax.set_ylabel("Count")
        ax.set_xlabel("Risk Level")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # ── Row 2: Glucose histogram | BMI histogram ──────────────────────────────
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown("##### Glucose Distribution")
        fig, ax = plt.subplots()
        ax.hist(raw_df["Glucose"], bins=20, color="#3498db", edgecolor="white")
        ax.set_title("Glucose Distribution")
        ax.set_xlabel("Glucose (mg/dL)")
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with r2c2:
        st.markdown("##### BMI Distribution")
        fig, ax = plt.subplots()
        ax.hist(raw_df["BMI"], bins=20, color="#9b59b6", edgecolor="white")
        ax.set_title("BMI Distribution")
        ax.set_xlabel("BMI (kg/m²)")
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # ── Row 3: Full-width feature correlation heatmap ─────────────────────────
    st.markdown("##### Feature Correlation Heatmap")
    numeric_cols = raw_df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        numeric_cols.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
