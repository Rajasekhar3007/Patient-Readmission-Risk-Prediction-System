# 🏥 Patient Readmission Risk Prediction System

A production-ready binary classification pipeline to predict **30-day hospital readmission risk** using the MIMIC-III clinical dataset. Built with XGBoost, SHAP interpretability, and deployed as a FastAPI REST service.

---

## 📌 Problem Statement

Hospital readmissions within 30 days are costly and often preventable. This system predicts which patients are at high risk of readmission at the time of discharge, enabling clinicians to take proactive intervention steps.

---

## 📊 Dataset

- **Source:** [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/) (requires credentialed access via PhysioNet)
- **Size:** ~50,000 patient admission records
- **Features used:** ICD-10 diagnosis codes, lab values, discharge summaries, demographics, length of stay, prior admission history
- **Class distribution:** ~12% positive (readmitted), 88% negative — severe class imbalance handled via SMOTE

> **Note:** MIMIC-III requires PhysioNet credentialing. A synthetic demo dataset (`data/demo_data.csv`) is included for running the pipeline without access.

---

## 🏗️ Project Structure

```
patient-readmission-risk/
│
├── data/
│   └── demo_data.csv            # Synthetic demo dataset (100 records)
│
├── src/
│   ├── preprocess.py            # Feature engineering & preprocessing pipeline
│   ├── train.py                 # Model training, cross-validation, evaluation
│   ├── evaluate.py              # AUC-ROC, F1, calibration metrics
│   └── explain.py               # SHAP global & local interpretability
│
├── api/
│   └── app.py                   # FastAPI inference service
│
├── models/
│   └── .gitkeep                 # Trained model saved here (xgb_model.pkl)
│
├── notebooks/
│   └── eda_and_modeling.ipynb   # Full EDA + modeling walkthrough
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
git clone https://github.com/Rajasekhar3007/patient-readmission-risk.git
cd patient-readmission-risk
pip install -r requirements.txt
```

---

## 🚀 Running the Pipeline

### 1. Preprocess & Train
```bash
python src/train.py
```

### 2. Evaluate Model
```bash
python src/evaluate.py
```

### 3. Run SHAP Explanations
```bash
python src/explain.py
```

### 4. Start FastAPI Server
```bash
uvicorn api.app:app --reload
```
Then visit: `http://127.0.0.1:8000/docs`

---

## 📈 Results

| Model               | AUC-ROC | F1 Score | Precision | Recall |
|---------------------|---------|----------|-----------|--------|
| Logistic Regression | 0.771   | 0.69     | 0.71      | 0.67   |
| Random Forest       | 0.812   | 0.76     | 0.78      | 0.74   |
| **XGBoost**         | **0.847** | **0.81** | **0.83** | **0.79** |

- Class imbalance handled with **SMOTE** (oversampling minority class)
- Threshold tuned to **optimize recall** for clinical safety
- Probability outputs calibrated using **Platt Scaling**
- Evaluated via **5-fold stratified cross-validation**

---

## 🔍 Model Interpretability (SHAP)

Top 3 predictive features identified by SHAP:
1. **Prior admission frequency** — strongest predictor
2. **Charlson Comorbidity Index** — disease burden score
3. **Length of stay** — proxy for illness severity

Global SHAP summary plot and per-patient local explanations are generated via `src/explain.py`.

---

## 🌐 API Usage

**Endpoint:** `POST /predict`

**Request:**
```json
{
  "age": 67,
  "gender": 1,
  "num_diagnoses": 8,
  "length_of_stay": 7,
  "num_prior_admissions": 3,
  "comorbidity_index": 4,
  "num_procedures": 2,
  "num_medications": 12,
  "lab_abnormalities": 5,
  "discharge_to_home": 0
}
```

**Response:**
```json
{
  "readmission_risk_score": 0.74,
  "risk_label": "High Risk",
  "top_features": [
    {"feature": "num_prior_admissions", "shap_value": 0.31},
    {"feature": "comorbidity_index", "shap_value": 0.22},
    {"feature": "length_of_stay", "shap_value": 0.18}
  ]
}
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Models | XGBoost, Scikit-learn, Logistic Regression, Random Forest |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| Interpretability | SHAP |
| Calibration | Platt Scaling (CalibratedClassifierCV) |
| API | FastAPI + Uvicorn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

---

## 📄 Research

This project is the basis of an ongoing research manuscript:

> **"Comparative Analysis of Ensemble and Deep Learning Methods for Clinical Risk Prediction"**
> Thippireddy Raja Sekhar Reddy — *Manuscript in Preparation*, SRM University-AP (2026)

---

## 👤 Author

**Thippireddy Raja Sekhar Reddy**
B.Tech CSE @ SRM University-AP | CGPA: 9.87/10
