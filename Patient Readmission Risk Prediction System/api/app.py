from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib
import shap
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import engineer_features

app = FastAPI(
    title="Patient Readmission Risk API",
    description="Predicts 30-day hospital readmission risk using XGBoost + SHAP interpretability.",
    version="1.0.0"
)

# ── Load model artifacts on startup ───────────────────────────────────────────
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "..", "models", "xgb_model.pkl")
SCALER_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")
IMPUTER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "imputer.pkl")

# Helper to load models dynamically or on-demand, fallback to lazy loading
model = None
scaler = None
imputer = None

def load_artifacts():
    global model, scaler, imputer
    if model is None and os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if scaler is None and os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    if imputer is None and os.path.exists(IMPUTER_PATH):
        imputer = joblib.load(IMPUTER_PATH)

# Try initial load
try:
    load_artifacts()
except Exception as e:
    print(f"[WARNING] Could not load model artifacts on startup: {e}")

FEATURE_COLS = [
    "age", "gender", "num_diagnoses", "length_of_stay",
    "num_prior_admissions", "comorbidity_index", "num_procedures",
    "num_medications", "lab_abnormalities", "discharge_to_home",
    "admission_burden", "complexity_score", "lab_per_day",
    "high_risk_discharge", "elderly"
]

THRESHOLD = 0.4


# ── Request / Response Schemas ─────────────────────────────────────────────────
class PatientInput(BaseModel):
    age:                  int   = Field(..., ge=0,  le=120,  json_schema_extra={"example": 67})
    gender:               int   = Field(..., ge=0,  le=1,    json_schema_extra={"example": 1})
    num_diagnoses:        int   = Field(..., ge=0,           json_schema_extra={"example": 8})
    length_of_stay:       int   = Field(..., ge=0,           json_schema_extra={"example": 7})
    num_prior_admissions: int   = Field(..., ge=0,           json_schema_extra={"example": 3})
    comorbidity_index:    int   = Field(..., ge=0,           json_schema_extra={"example": 4})
    num_procedures:       int   = Field(..., ge=0,           json_schema_extra={"example": 2})
    num_medications:      int   = Field(..., ge=0,           json_schema_extra={"example": 12})
    lab_abnormalities:    int   = Field(..., ge=0,           json_schema_extra={"example": 5})
    discharge_to_home:    int   = Field(..., ge=0,  le=1,    json_schema_extra={"example": 0})


class FeatureContribution(BaseModel):
    feature:    str
    shap_value: float


class PredictionResponse(BaseModel):
    readmission_risk_score: float
    risk_label:             str
    risk_threshold_used:    float
    top_features:           list[FeatureContribution]


# ── Helpers ────────────────────────────────────────────────────────────────────
def preprocess_input(patient: PatientInput) -> pd.DataFrame:
    # Handle Pydantic v1/v2 compatibility
    patient_dict = patient.model_dump() if hasattr(patient, 'model_dump') else patient.dict()
    df = pd.DataFrame([patient_dict])
    df = engineer_features(df)
    X  = df[FEATURE_COLS]
    X_imputed = imputer.transform(X)
    X_scaled  = scaler.transform(X_imputed)
    return pd.DataFrame(X_scaled, columns=FEATURE_COLS)


def get_shap_contributions(X_processed: pd.DataFrame, top_n: int = 5):
    try:
        base_model = model.calibrated_classifiers_[0].estimator
    except AttributeError:
        base_model = model

    explainer   = shap.TreeExplainer(base_model)
    shap_out    = explainer.shap_values(X_processed)
    
    # Extract values: SHAP can return a list for binary classes or a single array
    if isinstance(shap_out, list):
        # Index 1 is typically positive class, but fallback if only 1 class returned
        shap_values = shap_out[1][0] if len(shap_out) > 1 else shap_out[0][0]
    else:
        # XGBoost output array shape might be (samples, features) or (classes, samples, features)
        if len(shap_out.shape) == 3:
            shap_values = shap_out[1][0] if shap_out.shape[0] > 1 else shap_out[0][0]
        elif len(shap_out.shape) == 2:
            shap_values = shap_out[0]
        else:
            shap_values = shap_out

    contributions = sorted(
        zip(FEATURE_COLS, shap_values),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_n]

    return [FeatureContribution(feature=f, shap_value=round(float(v), 4))
            for f, v in contributions]


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Patient Readmission Risk API is running. Visit /docs for Swagger UI."}


@app.get("/health")
def health():
    load_artifacts()
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    load_artifacts()
    if model is None or scaler is None or imputer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run src/train.py first.")

    X_processed = preprocess_input(patient)
    risk_score  = float(model.predict_proba(X_processed)[0, 1])
    risk_label  = "High Risk" if risk_score >= THRESHOLD else "Low Risk"
    top_feats   = get_shap_contributions(X_processed)

    return PredictionResponse(
        readmission_risk_score=round(risk_score, 4),
        risk_label=risk_label,
        risk_threshold_used=THRESHOLD,
        top_features=top_feats
    )


@app.post("/predict/batch")
def predict_batch(patients: list[PatientInput]):
    load_artifacts()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    results = [predict(p) for p in patients]
    return {"predictions": results, "count": len(results)}
