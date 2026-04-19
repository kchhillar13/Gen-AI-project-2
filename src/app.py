import sys
from pathlib import Path

# Ensure Python always finds model.py and preprocess.py in the same src/ folder,
# even when the app is launched from outside the src/ directory.
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

# Import our custom preprocessing and evaluation functions
from preprocess import preprocess_uploaded_data
from model import evaluate_dataset

# ---------------------------------------------------------------------------
# PAGE CONFIGURATION - must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Patient Risk Assessment System",
    page_icon="🏥",
    layout="wide",
)

# ---------------------------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        :root {
            --primary-blue: #1a73e8;
            --deep-text: #172033;
            --muted-text: #5f6b7a;
            --soft-border: #e5eaf1;
            --card-bg: #ffffff;
        }

        .stApp {
            background: #ffffff;
            color: var(--deep-text);
        }

        .stApp,
        .stApp p,
        .stApp span,
        .stApp label,
        .stApp div,
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6 {
            color: var(--deep-text);
        }

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span {
            color: var(--deep-text);
        }

        section[data-testid="stSidebar"] {
            background: #f8fbff;
            border-right: 1px solid var(--soft-border);
        }

        section[data-testid="stSidebar"] * {
            color: var(--deep-text);
        }

        section[data-testid="stSidebar"] [role="radiogroup"] label {
            background: #ffffff;
            border: 1px solid var(--soft-border);
            border-radius: 8px;
            padding: 0.55rem 0.7rem;
            margin-bottom: 0.4rem;
            box-shadow: 0 3px 12px rgba(23, 32, 51, 0.05);
        }

        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
        }

        .app-header {
            border-left: 6px solid var(--primary-blue);
            padding: 1rem 1.2rem;
            margin-bottom: 1.2rem;
            background: linear-gradient(90deg, #f7fbff 0%, #ffffff 100%);
            box-shadow: 0 8px 24px rgba(26, 115, 232, 0.10);
        }

        .main-title {
            margin: 0;
            color: var(--deep-text);
            font-size: 2.25rem;
            font-weight: 800;
            letter-spacing: 0;
            line-height: 1.2;
        }

        .subtitle {
            margin: 0.35rem 0 0;
            color: var(--muted-text);
            font-size: 1rem;
            line-height: 1.5;
        }

        .section-card {
            background: var(--card-bg);
            border: 1px solid var(--soft-border);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 8px 22px rgba(23, 32, 51, 0.08);
            margin-bottom: 1rem;
            color: var(--deep-text);
        }

        .metric-card {
            background: var(--card-bg);
            border: 1px solid var(--soft-border);
            border-top: 4px solid var(--primary-blue);
            border-radius: 8px;
            padding: 1rem;
            min-height: 116px;
            box-shadow: 0 8px 22px rgba(23, 32, 51, 0.08);
            color: var(--deep-text);
        }

        .metric-label {
            color: var(--muted-text) !important;
            font-size: 0.88rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .metric-value {
            color: var(--deep-text) !important;
            font-size: 1.75rem;
            font-weight: 800;
            line-height: 1.2;
        }

        .metric-caption {
            color: var(--muted-text) !important;
            font-size: 0.78rem;
            margin-top: 0.35rem;
        }

        .sidebar-title {
            color: var(--deep-text);
            font-size: 1.25rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }

        .sidebar-note {
            color: var(--muted-text);
            font-size: 0.88rem;
            line-height: 1.45;
        }

        div[data-testid="stTabs"] button {
            background: #ffffff;
            color: var(--deep-text);
            font-weight: 700;
            border-radius: 8px 8px 0 0;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--primary-blue);
            border-bottom: 3px solid var(--primary-blue);
        }

        div[data-baseweb="select"] > div,
        div[data-testid="stExpander"] details,
        div[data-testid="stExpander"] summary {
            background: #ffffff;
            color: var(--deep-text);
            border-color: var(--soft-border);
        }

        div[data-testid="stExpander"] * {
            color: var(--deep-text);
        }

        div[data-testid="stDownloadButton"] button,
        div[data-testid="stButton"] button {
            background: var(--primary-blue);
            color: #ffffff !important;
            border-radius: 8px;
            border: 1px solid var(--primary-blue);
            font-weight: 800;
        }

        div[data-testid="stDownloadButton"] button p,
        div[data-testid="stButton"] button p {
            color: #ffffff !important;
        }

        div[data-testid="stDataFrame"] {
            background: #ffffff;
            border: 1px solid var(--soft-border);
            border-radius: 8px;
            box-shadow: 0 8px 22px rgba(23, 32, 51, 0.06);
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
            border: 1px solid var(--soft-border);
        }

        div[data-testid="stAlert"] * {
            color: var(--deep-text);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

REQUIRED_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

INVALID_FILE_MESSAGE = (
    "❌ Bundled dataset configuration error. The selected dataset must contain "
    "these 8 columns: Pregnancies, Glucose, BloodPressure, SkinThickness, "
    "Insulin, BMI, DiabetesPedigreeFunction, Age."
)


def metric_card(label, value, caption=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_level(probability):
    if probability >= 0.7:
        return "🔴 High"
    if probability >= 0.4:
        return "🟡 Medium"
    return "🟢 Low"


def validate_required_columns(df):
    return all(column in df.columns for column in REQUIRED_COLUMNS)


def build_predictions(raw_df):
    X_scaled, y_true = preprocess_uploaded_data(raw_df)
    results = evaluate_dataset(X_scaled, y_true)

    output_df = raw_df.copy().reset_index(drop=True)
    output_df["Prediction"] = [
        "🔴 Diabetes Risk" if prediction == 1 else "🟢 No Diabetes"
        for prediction in results["predictions"]
    ]
    probabilities = results["probabilities"]
    output_df["Risk Probability %"] = (probabilities * 100).round(1)
    output_df["Confidence"] = [
        f"{(prob if prob >= 0.5 else 1 - prob) * 100:.1f}%"
        for prob in probabilities
    ]
    output_df["Risk Level"] = [risk_level(prob) for prob in probabilities]

    risk_summary = {
        "Total Patients": len(output_df),
        "High Risk": int((output_df["Risk Level"] == "🔴 High").sum()),
        "Medium Risk": int((output_df["Risk Level"] == "🟡 Medium").sum()),
        "Low Risk": int((output_df["Risk Level"] == "🟢 Low").sum()),
    }

    return output_df, risk_summary, results, y_true


def style_risk_badges(df):
    def style_risk_column(value):
        if value == "🔴 High":
            return "background-color: #fde7e9; color: #b42318; font-weight: 700;"
        if value == "🟡 Medium":
            return "background-color: #fff4d6; color: #8a5a00; font-weight: 700;"
        if value == "🟢 Low":
            return "background-color: #e7f6ec; color: #0f7b3f; font-weight: 700;"
        return ""

    return df.style.map(style_risk_column, subset=["Risk Level"])


def render_glucose_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["Glucose"], bins=20, color="#1a73e8", edgecolor="white")
    ax.set_title("Glucose Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Glucose")
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", alpha=0.18)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_bmi_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["BMI"], bins=20, color="#34a853", edgecolor="white")
    ax.set_title("BMI Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("BMI")
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", alpha=0.18)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1 class="main-title">🏥 Intelligent Patient Risk Assessment System</h1>
        <p class="subtitle">
            Professional diabetes risk screening with model diagnostics,
            patient-level predictions, and an AI analysis workspace.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "⚠️ **Disclaimer:** This tool is for educational purposes only and is not a "
    "substitute for professional medical advice. Always consult a qualified healthcare provider."
)

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">🏥 Patient Risk AI</div>', unsafe_allow_html=True)

    data_source = st.radio(
        "Data source",
        options=[
            "🔴 High Risk Sample (20 patients)",
            "🟢 Low Risk Sample (20 patients)",
        ],
    )

    st.markdown("**About**")
    st.markdown(
        '<div class="sidebar-note">This app evaluates preloaded clinical sample datasets with a trained diabetes risk model.</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    [
        "📊 Model Dashboard",
        "🔍 Patient Predictions",
        "🤖 Agentic Analysis",
    ]
)

# ---------------------------------------------------------------------------
# TAB 1 - MODEL DASHBOARD
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Model Dashboard")

    with st.spinner("Processing model dashboard metrics..."):
        dashboard_df = pd.read_csv(BASE_DIR / "data" / "diabetes.csv")
        dashboard_X_scaled, dashboard_y_true = preprocess_uploaded_data(dashboard_df)
        dashboard_results = evaluate_dataset(dashboard_X_scaled, dashboard_y_true)
        dashboard_f1 = f1_score(dashboard_y_true, dashboard_results["predictions"])

    if dashboard_results["metrics_available"]:
        metric_cols = st.columns(4)
        with metric_cols[0]:
            metric_card("Accuracy", f"{dashboard_results['accuracy'] * 100:.2f}%", "Correct classifications")
        with metric_cols[1]:
            metric_card("F1", f"{dashboard_f1:.4f}", "Balance of precision and recall")
        with metric_cols[2]:
            metric_card("RMSE", f"{dashboard_results['rmse']:.4f}", "Root mean square error")
        with metric_cols[3]:
            metric_card("MAE", f"{dashboard_results['mae']:.4f}", "Mean absolute error")

        metric_cols = st.columns(4)
        with metric_cols[0]:
            metric_card("R²", f"{dashboard_results['r2']:.4f}", "Variance explained")

        st.markdown("### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        cm = confusion_matrix(dashboard_y_true, dashboard_results["predictions"])
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
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("### Feature Insights")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            render_glucose_distribution(dashboard_df)
            st.markdown("</div>", unsafe_allow_html=True)
        with chart_col2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            render_bmi_distribution(dashboard_df)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Model metrics are unavailable because the dashboard dataset has no Outcome column.")

# ---------------------------------------------------------------------------
# TAB 2 - PATIENT PREDICTIONS
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Patient Predictions")

    raw_df = None

    if data_source == "🔴 High Risk Sample (20 patients)":
        with st.spinner("Loading high risk sample data..."):
            raw_df = pd.read_csv(BASE_DIR / "data" / "sample_high_risk.csv")
    elif data_source == "🟢 Low Risk Sample (20 patients)":
        with st.spinner("Loading low risk sample data..."):
            raw_df = pd.read_csv(BASE_DIR / "data" / "sample_low_risk.csv")

    if raw_df is not None:
        if not validate_required_columns(raw_df):
            st.error(INVALID_FILE_MESSAGE)
            st.stop()

        try:
            with st.spinner("Running patient risk predictions..."):
                output_df, risk_summary, _, _ = build_predictions(raw_df)
        except Exception as exc:
            st.error(f"❌ An unexpected error occurred while processing this data: {exc}")
            st.stop()

        st.session_state["predictions_df"] = output_df
        st.session_state["risk_summary"] = risk_summary

        stat_cols = st.columns(4)
        with stat_cols[0]:
            metric_card("Total Patients", risk_summary["Total Patients"], "Records processed")
        with stat_cols[1]:
            metric_card("High Risk", risk_summary["High Risk"], "Needs priority review")
        with stat_cols[2]:
            metric_card("Medium Risk", risk_summary["Medium Risk"], "Monitor closely")
        with stat_cols[3]:
            metric_card("Low Risk", risk_summary["Low Risk"], "Lower estimated risk")

        st.markdown("### Full Results")
        display_cols = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
            "Prediction",
            "Risk Probability %",
            "Confidence",
            "Risk Level",
        ]
        st.dataframe(
            style_risk_badges(output_df[display_cols]),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Raw input data", expanded=False):
            st.dataframe(raw_df, use_container_width=True, hide_index=True)

        csv_bytes = output_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download results as CSV",
            data=csv_bytes,
            file_name="risk_assessment_results.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# TAB 3 - AGENTIC ANALYSIS
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Agentic Analysis")

    import json
    import os

    predictions_df = st.session_state.get("predictions_df")

    if predictions_df is None or predictions_df.empty:
        st.warning(
            "⚠️ No patient data loaded. Please go to the **Predictions tab** "
            "and select a sample dataset first."
        )
        st.button("Go to Predictions Tab")
    else:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            try:
                groq_api_key = st.secrets.get("GROQ_API_KEY")
            except Exception:
                groq_api_key = None

        if not groq_api_key:
            st.error(
                "🔑 GROQ_API_KEY is not configured. Set it in your "
                "environment or Streamlit secrets to enable AI analysis."
            )
            st.stop()

        try:
            from agent import run_health_agent
        except ImportError:
            st.error("agent.py not found. Please complete Phase 3 first.")
            st.stop()

        def format_probability(value):
            try:
                probability = float(value)
            except (TypeError, ValueError):
                return "N/A"

            if probability <= 1:
                probability *= 100
            return f"{probability:.0f}%"

        patient_labels = []
        for patient_idx, patient_row in predictions_df.reset_index(drop=True).iterrows():
            risk_level_label = patient_row.get("Risk Level", "Risk Unknown")
            risk_probability_pct = format_probability(patient_row.get("Risk Probability %"))
            patient_labels.append(
                f"Patient {patient_idx + 1} — {risk_level_label} Risk ({risk_probability_pct})"
            )

        default_idx = int(st.session_state.get("selected_patient_idx", 0))
        if default_idx >= len(patient_labels):
            default_idx = 0

        selected_label = st.selectbox(
            "Select a patient to analyse:",
            options=patient_labels,
            index=default_idx,
        )
        selected_patient_idx = patient_labels.index(selected_label)
        if st.session_state.get("selected_patient_idx") != selected_patient_idx:
            st.session_state.pop("agent_report", None)
        st.session_state["selected_patient_idx"] = selected_patient_idx

        selected_patient = predictions_df.reset_index(drop=True).iloc[selected_patient_idx]

        with st.expander("📋 Clinical Input Values", expanded=False):
            clinical_values = selected_patient[REQUIRED_COLUMNS].to_frame(name="Value")
            st.dataframe(clinical_values, use_container_width=True)

        _, button_col, _ = st.columns([1, 2, 1])
        with button_col:
            generate_report = st.button(
                "🧠 Generate AI Health Report",
                use_container_width=True,
            )

        if generate_report:
            risk_probability_pct = float(selected_patient.get("Risk Probability %", 0))
            patient_dict = {
                "Pregnancies": selected_patient["Pregnancies"],
                "Glucose": selected_patient["Glucose"],
                "BloodPressure": selected_patient["BloodPressure"],
                "SkinThickness": selected_patient["SkinThickness"],
                "Insulin": selected_patient["Insulin"],
                "BMI": selected_patient["BMI"],
                "DiabetesPedigreeFunction": selected_patient["DiabetesPedigreeFunction"],
                "Age": selected_patient["Age"],
                "risk_probability": risk_probability_pct / 100,
                "risk_level": selected_patient.get("Risk Level", "Unknown"),
            }

            try:
                with st.spinner("🔍 Analysing patient risk profile..."):
                    st.session_state["agent_report"] = run_health_agent(patient_dict)
            except Exception as exc:
                st.error(f"❌ Failed to generate AI health report: {exc}")

        report = st.session_state.get("agent_report")
        if report:
            risk_level = report.get("risk_level", selected_patient.get("Risk Level", "Unknown"))
            risk_probability_pct = report.get(
                "risk_probability_pct",
                format_probability(selected_patient.get("Risk Probability %")),
            )
            risk_level_text = str(risk_level)

            if "High" in risk_level_text:
                banner_bg = "#fde7e9"
                banner_border = "#d93025"
                banner_text = "#8c1d18"
            elif "Medium" in risk_level_text:
                banner_bg = "#fff4d6"
                banner_border = "#fbbc04"
                banner_text = "#7a4f00"
            else:
                banner_bg = "#e7f6ec"
                banner_border = "#34a853"
                banner_text = "#0b6b35"

            st.markdown(
                f"""
                <div style="
                    background: {banner_bg};
                    border-left: 6px solid {banner_border};
                    border-radius: 8px;
                    padding: 1.1rem 1.25rem;
                    margin: 1rem 0;
                    color: {banner_text};
                    box-shadow: 0 8px 22px rgba(23, 32, 51, 0.08);
                ">
                    <div style="font-size: 0.95rem; font-weight: 700;">Risk Assessment</div>
                    <div style="font-size: 2rem; font-weight: 800; line-height: 1.2;">
                        {risk_level_text} · {risk_probability_pct}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("### Key Risk Factors")
            risk_factors = report.get("key_risk_factors", report.get("risk_factors", []))
            if isinstance(risk_factors, str):
                risk_factors = [risk_factors]
            factor_cols = st.columns(3)
            for card_idx in range(3):
                factor_text = risk_factors[card_idx] if card_idx < len(risk_factors) else "Not specified"
                with factor_cols[card_idx]:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">⚠️ Risk Factor {card_idx + 1}</div>
                            <div style="color: #172033; font-size: 1rem; font-weight: 700; line-height: 1.45;">
                                {factor_text}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("### 📝 Clinical Analysis")
            st.info(report.get("risk_explanation", "No clinical analysis was returned."))

            recommendations = report.get("recommendations", {})
            st.markdown("### Recommendations")

            st.markdown("#### 🥗 Lifestyle")
            lifestyle_items = recommendations.get("lifestyle", [])
            if isinstance(lifestyle_items, str):
                lifestyle_items = [lifestyle_items]
            for item in lifestyle_items:
                st.markdown(f"- {item}")

            st.markdown("#### 💊 Medication Note")
            st.warning(recommendations.get("medication_note", "No medication note was returned."))

            st.markdown("#### 📅 Follow-up")
            st.success(recommendations.get("follow_up", "No follow-up guidance was returned."))

            with st.expander("📚 Medical Guideline Sources Used", expanded=False):
                sources = report.get("guideline_sources", [])
                if isinstance(sources, str):
                    sources = [sources]
                for source in sources:
                    st.markdown(f"📖 {source}")

            st.error(report.get("disclaimer", "This AI report is educational and is not a substitute for professional medical advice."))

            st.download_button(
                "⬇️ Download Full Report (JSON)",
                data=json.dumps(report, indent=2),
                file_name=f"health_report_patient_{selected_patient_idx + 1}.json",
                mime="application/json",
            )
