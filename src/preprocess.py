import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# BASE_DIR always points to the project root (two levels up from src/)
BASE_DIR = Path(__file__).resolve().parent.parent


def preprocess_data():
    """
    Full preprocessing pipeline for the Pima Indians Diabetes Dataset.
    Returns: X_train, X_test, y_train, y_test (as NumPy arrays)
    """

    # ------------------------------------------------------------------
    # STEP 1 — Load the dataset from the CSV file
    # We read the CSV into a pandas DataFrame and print a quick overview.
    # ------------------------------------------------------------------
    df = pd.read_csv(BASE_DIR / "data" / "diabetes.csv")
    print("=== First 5 rows of the dataset ===")
    print(df.head())
    print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    # ------------------------------------------------------------------
    # STEP 2 — Handle missing / zero values
    # In this dataset, zeros in certain medical columns are biologically
    # impossible (e.g. a person cannot have 0 blood pressure or 0 glucose).
    # We treat those zeros as missing data by replacing them with NaN,
    # then fill each NaN with the median of its column.
    # Median is preferred over mean because it is more robust to outliers.
    # ------------------------------------------------------------------
    zero_as_null_cols = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]

    # Replace 0 with NaN in the specified columns
    df[zero_as_null_cols] = df[zero_as_null_cols].replace(0, np.nan)

    # Fill each NaN with the median value of that column
    for col in zero_as_null_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  '{col}' — NaN values filled with median: {median_val:.2f}")

    print()

    # ------------------------------------------------------------------
    # STEP 3 — Separate features (X) and the target label (y)
    # X contains all columns except 'Outcome'.
    # y contains only the 'Outcome' column (0 = No Diabetes, 1 = Diabetes).
    # ------------------------------------------------------------------
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    print(f"Feature matrix X shape : {X.shape}")
    print(f"Target vector   y shape : {y.shape}\n")

    # ------------------------------------------------------------------
    # STEP 4 — Feature scaling using StandardScaler
    # StandardScaler transforms each feature so it has mean=0 and std=1.
    # This ensures that no single feature dominates due to its scale.
    # We fit the scaler on ALL features and then save it to 'scaler.pkl'
    # so the same transformation can be applied to new data at prediction time.
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the fitted scaler for later use in the Streamlit app / model
    joblib.dump(scaler, BASE_DIR / "models" / "scaler.pkl")
    print(f"Fitted StandardScaler saved to '{BASE_DIR / 'models' / 'scaler.pkl'}'")

    # ------------------------------------------------------------------
    # STEP 5 — Split data into training (80%) and testing (20%) sets
    # random_state=42 ensures reproducible results every time the script runs.
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42
    )

    print(f"\nTraining set size : {X_train.shape[0]} samples")
    print(f"Testing  set size : {X_test.shape[0]} samples\n")

    # ------------------------------------------------------------------
    # STEP 6 — Return the split datasets
    # ------------------------------------------------------------------
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# NEW FUNCTION — preprocess_uploaded_data(df)
# This function is used by app.py when a user uploads their own CSV/Excel file.
# Unlike preprocess_data() which loads a fixed file and fits a new scaler,
# this function accepts an already-loaded DataFrame and applies the EXISTING
# trained scaler (scaler.pkl) so predictions are consistent with the model.
# ---------------------------------------------------------------------------

# The 8 feature columns the model was trained on (order matters for scaling)
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

# Columns where 0 is biologically impossible and should be treated as missing
ZERO_AS_NULL_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def preprocess_uploaded_data(df):
    """
    Preprocesses a user-uploaded pandas DataFrame for inference.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw DataFrame from an uploaded CSV or Excel file.

    Returns
    -------
    X_scaled : np.ndarray
        Scaled feature matrix ready for model.predict()
    y : pandas.Series or None
        True labels if 'Outcome' column is present, else None.
    """

    # ------------------------------------------------------------------
    # STEP 1 — Validate that all required feature columns are present
    # If any column is missing, raise a ValueError immediately with a clear
    # message listing exactly which columns are absent.
    # ------------------------------------------------------------------
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Uploaded file is missing the following required columns: "
            f"{missing_cols}. "
            f"Please ensure your file has all 8 feature columns: {REQUIRED_COLUMNS}"
        )

    # ------------------------------------------------------------------
    # STEP 2 — Check for the optional Outcome column (true labels)
    # If present, we separate it so we can later compare predictions vs truth.
    # If absent, we set y = None — predictions can still be made without it.
    # ------------------------------------------------------------------
    if "Outcome" in df.columns:
        y = df["Outcome"].reset_index(drop=True)
    else:
        y = None

    # Extract only the 8 feature columns in the correct order
    X = df[REQUIRED_COLUMNS].copy()

    # ------------------------------------------------------------------
    # STEP 3 — Handle missing / zero values
    # Replace biologically impossible zeros with NaN, then fill each NaN
    # with the median of that column in the uploaded data.
    # ------------------------------------------------------------------
    X[ZERO_AS_NULL_COLS] = X[ZERO_AS_NULL_COLS].replace(0, np.nan)

    for col in ZERO_AS_NULL_COLS:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)

    # ------------------------------------------------------------------
    # STEP 4 — Scale features using the EXISTING trained scaler
    # We load scaler.pkl (fitted during model training) and call .transform()
    # NOT .fit_transform() — reusing the original scaler is critical to ensure
    # the uploaded data is scaled on the same axis as the training data.
    # ------------------------------------------------------------------
    scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")
    X_scaled = scaler.transform(X)

    # ------------------------------------------------------------------
    # STEP 5 — Return scaled features and labels (labels may be None)
    # ------------------------------------------------------------------
    return X_scaled, y


# Allow this file to be run directly for a quick sanity-check
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()
    print("Preprocessing complete ✓")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
