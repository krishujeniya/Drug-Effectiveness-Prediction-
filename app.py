"""Streamlit conversion of drug-effectiveness-prediction-s.ipynb.
Run: streamlit run app.py
Dataset: real_drug_dataset.csv from the Kaggle notebook's data source.
"""
import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
FEATURES = ["Age", "Gender", "Condition", "Drug_Name", "Dosage_mg", "Treatment_Duration_days"]
NUMERIC = ["Age", "Dosage_mg", "Treatment_Duration_days"]
CATEGORICAL = ["Gender", "Condition", "Drug_Name"]
TARGET = "Improvement_Score"
REQUIRED = FEATURES + [TARGET]

st.set_page_config(page_title="Drug Effectiveness Prediction", page_icon="💊", layout="wide")
st.title("💊 Drug Effectiveness Prediction")
st.caption("Streamlit app converted from the Kaggle notebook by krishujeniya")
st.warning("Research/demo only. Predictions are not validated for clinical use and must not guide medication selection or dosing.")

@st.cache_data(show_spinner=False)
def read_data(raw):
    return pd.read_csv(io.BytesIO(raw))

@st.cache_resource(show_spinner="Training and tuning Gradient Boosting (this may take several minutes)…")
def train_model(raw):
    df = read_data(raw).copy()
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        raise ValueError("Dataset missing columns: " + ", ".join(missing_cols))
    df = df.dropna(subset=REQUIRED)
    if len(df) < 30:
        raise ValueError("At least 30 complete rows are required.")
    for col in NUMERIC + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=NUMERIC + [TARGET])
    if len(df) < 30:
        raise ValueError("Not enough valid numeric records remain.")
    for col in CATEGORICAL:
        df[col] = df[col].astype(str)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    model = Pipeline([("prep", preprocess), ("model", GradientBoostingRegressor(random_state=SEED))])
    param_dist = {
        "model__n_estimators": [50, 100, 150, 200, 250, 300, 400, 500, 700, 1000],
        "model__learning_rate": [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
        "model__max_depth": [1, 2, 2, 3, 3, 3, 4, 4, 5, 6],
        "model__min_samples_leaf": [1, 1, 2, 3, 4, 5, 7, 10, 15, 20],
        "model__min_samples_split": [2, 4, 6, 8, 10, 12, 15, 20, 25, 30],
        "model__subsample": [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.0],
    }
    search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=40,
        cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring="neg_mean_absolute_error", random_state=SEED, n_jobs=-1)
    search.fit(X_train, y_train)
    pred = search.best_estimator_.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R²": r2_score(y_test, pred),
        "CV MAE": -search.best_score_,
    }
    results = X_test.copy()
    results["actual"] = y_test
    results["predicted"] = np.round(pred, 2)
    results["residual"] = results["actual"] - results["predicted"]
    valid_drugs = {str(k): sorted(map(str, v)) for k, v in df.groupby("Condition")["Drug_Name"].unique().items()}
    return search.best_estimator_, metrics, results, valid_drugs, df, search.best_params_

st.sidebar.header("Dataset")
upload = st.sidebar.file_uploader("Upload real_drug_dataset.csv", type="csv")
st.sidebar.info("The original Kaggle dataset is not embedded in the notebook. Upload its CSV to enable training and predictions.")
if upload is None:
    st.info("👈 Upload the dataset CSV in the sidebar to start.")
    st.markdown("**Required columns:** " + ", ".join(f"`{x}`" for x in REQUIRED))
    st.stop()

raw = upload.getvalue()
try:
    model, metrics, results, valid_drugs, df, best_params = train_model(raw)
except Exception as exc:
    st.error(f"Could not load or train on the dataset: {exc}")
    st.stop()

predict_tab, evaluation_tab, data_tab = st.tabs(["🔮 Predict", "📊 Model evaluation", "📋 Dataset"])
with predict_tab:
    st.subheader("Predict improvement score")
    st.write("Choose a condition and a drug that appears with that condition in the uploaded dataset.")
    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=45)
            genders = sorted(df["Gender"].unique().tolist())
            gender = st.selectbox("Gender", genders)
            condition = st.selectbox("Condition", sorted(valid_drugs))
        with right:
            drug = st.selectbox("Drug", valid_drugs[condition])
            dosage = st.number_input("Dosage (mg)", min_value=0.0, value=500.0, step=10.0)
            duration = st.number_input("Treatment duration (days)", min_value=1, max_value=3650, value=45)
        submitted = st.form_submit_button("Predict effectiveness", type="primary")
    if submitted:
        patient = pd.DataFrame([{
            "Age": age, "Gender": gender, "Condition": condition,
            "Drug_Name": drug, "Dosage_mg": dosage,
            "Treatment_Duration_days": duration,
        }])
        prediction = float(model.predict(patient)[0])
        st.metric("Predicted improvement score (0–10 scale)", f"{prediction:.2f} / 10")
        if prediction < 0 or prediction > 10:
            st.caption("Note: the regressor can predict values outside the dataset's nominal 0–10 scale.")
        st.warning("The notebook found little predictive signal in these features. Do not interpret this score as a reliable estimate of an individual's treatment response.")

with evaluation_tab:
    st.subheader("Held-out test performance")
    a, b, c, d = st.columns(4)
    a.metric("MAE", f"{metrics['MAE']:.3f}")
    b.metric("RMSE", f"{metrics['RMSE']:.3f}")
    c.metric("R²", f"{metrics['R²']:.3f}")
    d.metric("Cross-validation MAE", f"{metrics['CV MAE']:.3f}")
    st.caption("80/20 train-test split, 5-fold cross-validation, 40 randomized Gradient Boosting parameter settings (as in the notebook).")
    st.subheader("Actual vs. predicted")
    st.scatter_chart(results, x="actual", y="predicted")
    st.subheader("Residual distribution")
    st.bar_chart(results["residual"].round(1).value_counts().sort_index())
    st.subheader("Best hyperparameters")
    st.json(best_params)
    st.download_button("Download test predictions CSV", results.to_csv(index=False).encode("utf-8"), "test_predictions.csv", "text/csv")

with data_tab:
    st.subheader("Uploaded dataset")
    st.write(f"{len(df):,} usable patient records")
    st.dataframe(df, use_container_width=True)
    st.subheader("Improvement score distribution")
    st.bar_chart(df[TARGET].round(1).value_counts().sort_index())
