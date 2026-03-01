import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import our custom preprocessing pipeline
from preprocess import preprocess_data


# ------------------------------------------------------------------
# STEP 1 — Load preprocessed data
# We call preprocess_data() which handles loading the CSV, cleaning
# zero/null values, scaling features, and splitting into train/test sets.
# ------------------------------------------------------------------
def train_model():
    """
    Trains a Logistic Regression model on the diabetes dataset.
    Evaluates performance and saves the trained model to 'model.pkl'.
    """

    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data()

    # ------------------------------------------------------------------
    # STEP 2 — Train the Logistic Regression model
    # max_iter=1000 gives the solver enough iterations to converge.
    # random_state=42 ensures reproducibility.
    # ------------------------------------------------------------------
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete ✓")

    # ------------------------------------------------------------------
    # STEP 3 — Evaluate the model on the test set
    # We print three metrics:
    #   • Accuracy     — overall percentage of correct predictions
    #   • Classification Report — Precision, Recall, F1 per class
    #   • Confusion Matrix      — True/False Positives and Negatives
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'='*45}")
    print(f"  MODEL EVALUATION RESULTS")
    print(f"{'='*45}")
    print(f"  Accuracy Score : {accuracy * 100:.2f}%")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))
    print(f"  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  {'':16} Predicted No  Predicted Yes")
    print(f"  Actual No      {cm[0][0]:>12}  {cm[0][1]:>13}")
    print(f"  Actual Yes     {cm[1][0]:>12}  {cm[1][1]:>13}")
    print(f"{'='*45}\n")

    # ------------------------------------------------------------------
    # STEP 4 — Save the trained model to disk
    # We use joblib.dump() to serialize the model object to 'model.pkl'.
    # This file will be loaded later by the Streamlit app for predictions.
    # ------------------------------------------------------------------
    joblib.dump(model, "model.pkl")
    print("Trained model saved to 'model.pkl' ✓")

    return model


# ------------------------------------------------------------------
# STEP 5 — Prediction function for new patient data
# This function is called by the Streamlit frontend (app.py).
# It accepts a dictionary of raw patient values, scales them using
# the saved scaler, then returns the predicted class and probability.
# ------------------------------------------------------------------

# The exact column order that the model was trained on
FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def predict_risk(input_data: dict):
    """
    Predicts diabetes risk for a single patient.

    Parameters
    ----------
    input_data : dict
        Keys must match FEATURE_COLUMNS exactly.
        Example:
            {
                "Pregnancies": 2,
                "Glucose": 138,
                "BloodPressure": 70,
                "SkinThickness": 28,
                "Insulin": 100,
                "BMI": 30.5,
                "DiabetesPedigreeFunction": 0.45,
                "Age": 34
            }

    Returns
    -------
    prediction : int
        0 = No Diabetes, 1 = Diabetes
    probability : float
        Confidence score for the positive (Diabetes) class (0.0 – 1.0)
    """

    # Load the saved model and scaler from disk
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")

    # Convert the input dictionary to a numpy array in the correct column order
    raw_values = np.array([[input_data[col] for col in FEATURE_COLUMNS]])

    # Scale the input the same way the training data was scaled
    scaled_values = scaler.transform(raw_values)

    # Make prediction and get probability of the positive class (class 1)
    prediction = int(model.predict(scaled_values)[0])
    probability = float(model.predict_proba(scaled_values)[0][1])

    return prediction, probability


# ------------------------------------------------------------------
# STEP 6 — Main block
# When this file is run directly (`python model.py`), it automatically
# triggers the full training + evaluation + saving pipeline.
# ------------------------------------------------------------------
if __name__ == "__main__":
    train_model()
