# 🏥 Intelligent Patient Risk Assessment System

![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📌 Project Description

The **Intelligent Patient Risk Assessment System** is a machine learning–powered web application that predicts the likelihood of a patient having diabetes based on key medical vitals and demographic information. It uses a **Logistic Regression** model trained on the well-known Pima Indians Diabetes Dataset to deliver instant, probability-based risk assessments. The application is built with **Streamlit**, providing an intuitive and interactive interface that requires no technical knowledge to operate. This project demonstrates the end-to-end pipeline of an ML system — from data preprocessing and model training through to a fully deployed web interface.

---

## 🔗 Live Demo

🌐 **Live App:** [Insert Streamlit Cloud URL here]

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.9+** | Core programming language |
| **Scikit-Learn** | Machine learning model training and evaluation |
| **Pandas** | Data loading, manipulation, and analysis |
| **NumPy** | Numerical computations and array operations |
| **Streamlit** | Interactive web application frontend |
| **Matplotlib** | Plotting charts and visualisations |
| **Seaborn** | Confusion matrix heatmap and statistical plots |
| **Joblib** | Saving and loading the trained model and scaler |

---

## 📂 Dataset

This project uses the **Pima Indians Diabetes Dataset**, originally sourced from the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK).

| Attribute | Detail |
|---|---|
| **Total Records** | 768 patients |
| **Features** | 8 numeric input features |
| **Target Variable** | `Outcome` — binary (0 = No Diabetes, 1 = Diabetes) |
| **Task Type** | Binary Classification |

### Features

| # | Feature | Description |
|---|---|---|
| 1 | `Pregnancies` | Number of times pregnant |
| 2 | `Glucose` | Plasma glucose concentration (mg/dL) |
| 3 | `BloodPressure` | Diastolic blood pressure (mm Hg) |
| 4 | `SkinThickness` | Triceps skinfold thickness (mm) |
| 5 | `Insulin` | 2-hour serum insulin (µU/mL) |
| 6 | `BMI` | Body mass index (kg/m²) |
| 7 | `DiabetesPedigreeFunction` | Diabetes likelihood based on family history |
| 8 | `Age` | Patient age (years) |

> **Data file:** Place `diabetes.csv` inside the `data/` folder before running the app.

---

## 📁 Project Structure

```
patient-risk-assessment/
│
├── app.py              # Streamlit web app — the visual frontend
├── model.py            # ML model training, evaluation, and predict_risk() function
├── preprocess.py       # Data loading, cleaning, scaling, and train/test split
├── requirements.txt    # All Python dependencies with version pins
├── README.md           # Project documentation (this file)
│
├── data/
│   └── diabetes.csv    # Pima Indians Diabetes Dataset (add manually)
│
├── model.pkl           # Serialised trained Logistic Regression model (auto-generated)
└── scaler.pkl          # Fitted StandardScaler object (auto-generated)
```

---

## 🚀 How to Run Locally

Follow these steps in your terminal, in order:

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/patient-risk-assessment.git
cd patient-risk-assessment
```

### 2. Create and activate a Python virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install all dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Download `diabetes.csv` and place it inside the `data/` folder:
```
patient-risk-assessment/data/diabetes.csv
```

### 5. Train and save the ML model
```bash
python model.py
```
This generates `model.pkl` and `scaler.pkl` in the project root.

### 6. Launch the Streamlit app
```bash
streamlit run app.py
```
The app will open automatically at **http://localhost:8501**.

---

## 🧠 ML Methodology

### Data Preprocessing
The raw dataset contains biologically impossible zero values in columns such as `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` — these represent missing data. All such zeros were replaced with `NaN` and then imputed with the **column median** (preferred over the mean due to its robustness against outliers). Features were then standardised using **`StandardScaler`** (zero mean, unit variance) to ensure no single feature dominates due to scale differences. The dataset was split **80% training / 20% testing** with `random_state=42` for full reproducibility.

### Model Choice — Logistic Regression
**Logistic Regression** was selected as the baseline model for this classification task because:
- It is highly interpretable and well-suited to binary classification problems.
- It produces calibrated probability outputs, allowing the app to show a meaningful **risk percentage**.
- It trains quickly and generalises well on moderately sized, tabular medical datasets.
- It serves as an excellent benchmark before exploring more complex models (e.g. Random Forest, XGBoost).

### Evaluation Metrics
| Metric | Why it was used |
|---|---|
| **Accuracy Score** | Overall correctness of predictions |
| **Precision & Recall** | Critical for imbalanced medical data — minimising false negatives (missed diagnoses) |
| **F1-Score** | Harmonic mean of Precision and Recall; balances both concerns |
| **Confusion Matrix** | Visual breakdown of TP, TN, FP, FN across both classes |

---

## 👥 Team

| Name | Role |
|---|---|
| Team Member 1 | ML Model Development & Data Preprocessing |
| Team Member 2 | Streamlit Frontend & UI/UX Design |
| Team Member 3 | Documentation, Testing & Deployment |

---

## ⚠️ Disclaimer

> This application is developed **strictly for educational and academic purposes**.  
> It is **not** a medical device and should **not** be used as a substitute for professional medical diagnosis, advice, or treatment.  
> Always consult a qualified and licensed healthcare professional for any medical concerns.  
> The authors accept no liability for decisions made based on the output of this system.
