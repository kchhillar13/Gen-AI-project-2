import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# Import our trained model's prediction function and preprocessing pipeline
from model import predict_risk, train_model, FEATURE_COLUMNS
from preprocess import preprocess_data

# ------------------------------------------------------------------
# PAGE CONFIGURATION
# Must be the very first Streamlit call in the script.
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Patient Risk Assessment System",
    page_icon="🏥",
    layout="wide",
)

# ------------------------------------------------------------------
# CUSTOM CSS — gives the app a cleaner, more professional look
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 700;
            color: #1f3d6b;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            font-size: 1.05rem;
            color: #555;
            margin-bottom: 1.2rem;
        }
        .result-box {
            padding: 1rem 1.4rem;
            border-radius: 10px;
            font-size: 1.1rem;
            font-weight: 600;
        }
        .section-divider {
            border-top: 1px solid #e0e0e0;
            margin: 1.5rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# HEADER SECTION
# ------------------------------------------------------------------
st.markdown('<p class="main-title">🏥 Intelligent Patient Risk Assessment System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter patient details below to assess diabetes risk using a trained Machine Learning model.</p>', unsafe_allow_html=True)

# Disclaimer shown prominently at the top
st.warning(
    "⚠️ **Disclaimer:** This tool is for educational purposes only and is "
    "not a substitute for professional medical advice. Always consult a "
    "qualified healthcare provider for diagnosis and treatment."
)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# TWO-COLUMN LAYOUT
# Left  → Input form
# Right → Prediction results
# ------------------------------------------------------------------
left_col, right_col = st.columns([1.1, 1], gap="large")

# ------------------------------------------------------------------
# LEFT COLUMN — Patient Input Form
# ------------------------------------------------------------------
with left_col:
    st.subheader("📋 Patient Details")

    with st.form(key="patient_form"):

        # Row 1
        r1_c1, r1_c2 = st.columns(2)
        with r1_c1:
            pregnancies = st.number_input(
                "Pregnancies",
                min_value=0, max_value=20, value=1, step=1,
                help="Number of times pregnant",
            )
        with r1_c2:
            glucose = st.number_input(
                "Glucose (mg/dL)",
                min_value=0, max_value=300, value=120, step=1,
                help="Plasma glucose concentration (2-hour oral glucose tolerance test)",
            )

        # Row 2
        r2_c1, r2_c2 = st.columns(2)
        with r2_c1:
            blood_pressure = st.number_input(
                "Blood Pressure (mm Hg)",
                min_value=0, max_value=200, value=70, step=1,
                help="Diastolic blood pressure",
            )
        with r2_c2:
            skin_thickness = st.number_input(
                "Skin Thickness (mm)",
                min_value=0, max_value=100, value=20, step=1,
                help="Triceps skinfold thickness",
            )

        # Row 3
        r3_c1, r3_c2 = st.columns(2)
        with r3_c1:
            insulin = st.number_input(
                "Insulin (µU/mL)",
                min_value=0, max_value=900, value=80, step=1,
                help="2-hour serum insulin",
            )
        with r3_c2:
            bmi = st.number_input(
                "BMI (kg/m²)",
                min_value=0.0, max_value=70.0, value=25.0, step=0.1,
                format="%.1f",
                help="Body Mass Index",
            )

        # Row 4
        r4_c1, r4_c2 = st.columns(2)
        with r4_c1:
            dpf = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0, max_value=3.0, value=0.5, step=0.01,
                format="%.3f",
                help="Likelihood of diabetes based on family history",
            )
        with r4_c2:
            age = st.number_input(
                "Age (years)",
                min_value=1, max_value=120, value=30, step=1,
                help="Patient age in years",
            )

        # Submit button
        submitted = st.form_submit_button(
            "🔍 Assess Risk",
            use_container_width=True,
            type="primary",
        )

# ------------------------------------------------------------------
# RIGHT COLUMN — Prediction Results
# ------------------------------------------------------------------
with right_col:
    st.subheader("📊 Risk Assessment Result")

    if submitted:
        # Build the input dictionary matching FEATURE_COLUMNS order
        input_data = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
        }

        # Run the prediction using the saved model and scaler
        with st.spinner("Analysing patient data..."):
            prediction, probability = predict_risk(input_data)

        prob_percent = probability * 100

        # ----------------------------------------------------------
        # Display result based on prediction (0 = Low Risk, 1 = High Risk)
        # ----------------------------------------------------------
        if prediction == 1:
            # High Risk
            st.error(
                "🔴 **HIGH RISK** — This patient shows indicators strongly "
                "associated with diabetes."
            )
        else:
            # Low Risk
            st.success(
                "🟢 **LOW RISK** — This patient does not show strong "
                "indicators of diabetes."
            )

        # Show probability percentage and progress bar
        st.markdown(f"### Risk Probability: **{prob_percent:.1f}%**")
        st.progress(probability)

        # Additional breakdown metrics
        st.markdown("---")
        m1, m2 = st.columns(2)
        m1.metric("Prediction", "Diabetic" if prediction == 1 else "Non-Diabetic")
        m2.metric("Confidence", f"{prob_percent:.1f}%")

        # Key input summary
        with st.expander("🗒️ Input Summary", expanded=False):
            summary_data = {
                "Feature": list(input_data.keys()),
                "Value": list(input_data.values()),
            }
            import pandas as pd
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

    else:
        # Placeholder shown before the form is submitted
        st.info("👈 Fill in the patient details on the left and click **🔍 Assess Risk** to see the result.")
        st.image(
            "https://img.icons8.com/color/200/medical-doctor.png",
            width=180,
        )

# ------------------------------------------------------------------
# BOTTOM SECTION — Model Performance (collapsible)
# Reuses preprocess_data() to reconstruct X_test, y_test,
# then loads the model to compute live evaluation metrics.
# ------------------------------------------------------------------
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

with st.expander("📊 View Model Performance", expanded=False):
    st.markdown("These metrics reflect how the model performs on the **held-out test set (20% of data)**.")

    import joblib

    # Load the saved model and scaler
    model = joblib.load("model.pkl")

    # Rerun preprocessing to get the test split
    _, X_test, _, y_test = preprocess_data()

    # Generate predictions on the test set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # ---- Metrics row ----
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("✅ Accuracy", f"{acc * 100:.2f}%")
    col_b.metric("📦 Test Samples", len(y_test))
    col_c.metric("🔢 Features Used", "8")

    st.markdown("---")

    # ---- Confusion Matrix Heatmap ----
    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

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
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("Actual Label", fontsize=11)
    ax.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
