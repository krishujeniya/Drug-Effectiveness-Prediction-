"""Simple drug effectiveness predictor. Run with: streamlit run app.py
Keep real_drug_dataset.csv alongside this file in your GitHub repository.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Drug Effectiveness Predictor", page_icon="💊", layout="centered")

FEATURES = ["Age", "Gender", "Condition", "Drug_Name", "Dosage_mg", "Treatment_Duration_days"]
NUMERIC = ["Age", "Dosage_mg", "Treatment_Duration_days"]
CATEGORICAL = ["Gender", "Condition", "Drug_Name"]
TARGET = "Improvement_Score"
DATASET = Path(__file__).resolve().parent / "real_drug_dataset.csv"


@st.cache_resource(show_spinner="Preparing prediction model…")
def load_model():
    if not DATASET.is_file():
        raise FileNotFoundError("real_drug_dataset.csv is missing. Add it beside app.py in your repository.")
    data = pd.read_csv(DATASET)
    missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
    if missing:
        raise ValueError("Missing dataset columns: " + ", ".join(missing))
    data = data[FEATURES + [TARGET]].copy()
    for col in NUMERIC + [TARGET]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=FEATURES + [TARGET])
    if data.empty:
        raise ValueError("The dataset has no complete valid records.")
    for col in CATEGORICAL:
        data[col] = data[col].astype(str).str.strip()

    preprocessing = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    # Best Gradient Boosting hyperparameters reported by the original notebook.
    predictor = Pipeline([
        ("prep", preprocessing),
        ("model", GradientBoostingRegressor(
            random_state=42, subsample=0.6, n_estimators=150,
            min_samples_split=25, min_samples_leaf=4,
            max_depth=4, learning_rate=0.001,
        )),
    ])
    predictor.fit(data[FEATURES], data[TARGET])
    return predictor, data


st.title("💊 Drug Effectiveness Prediction")
st.write("Enter patient and treatment details to estimate an improvement score.")

try:
    model, data = load_model()
except Exception as exc:
    st.error(str(exc))
    st.stop()

conditions = sorted(data["Condition"].unique().tolist())
genders = sorted(data["Gender"].unique().tolist())

# The drug choices update automatically to match the selected condition.
with st.form("prediction", clear_on_submit=False):
    age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
    gender = st.selectbox("Gender", genders)
    condition = st.selectbox("Medical condition", conditions)
    available_drugs = sorted(data.loc[data["Condition"] == condition, "Drug_Name"].unique().tolist())
    drug = st.selectbox("Drug name", available_drugs)
    dosage = st.number_input("Dosage (mg)", min_value=0.0, value=100.0, step=10.0)
    duration = st.number_input("Treatment duration (days)", min_value=1, max_value=3650, value=7, step=1)
    predict = st.form_submit_button("Predict Effectiveness", type="primary", use_container_width=True)

if predict:
    entry = pd.DataFrame([{
        "Age": age, "Gender": gender, "Condition": condition,
        "Drug_Name": drug, "Dosage_mg": dosage,
        "Treatment_Duration_days": duration,
    }])
    score = float(np.clip(model.predict(entry)[0], 1, 10))
    # Assign levels using the nearest whole-number improvement score.
    rounded = int(np.floor(score + 0.5))
    effectiveness = "Low" if rounded <= 4 else "Medium" if rounded <= 7 else "High"
    st.success(f"Predicted Effectiveness: {effectiveness}")
    st.metric("Predicted Improvement Score", f"{score:.2f} / 10")
    st.caption("Low: 1–4  |  Medium: 5–7  |  High: 8–10")

st.caption("Experimental prediction only; not suitable for medical decisions. The original notebook reported limited predictive accuracy.")
