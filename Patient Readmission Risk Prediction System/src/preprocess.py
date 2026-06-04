import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import os

FEATURE_COLS = [
    "age", "gender", "num_diagnoses", "length_of_stay",
    "num_prior_admissions", "comorbidity_index", "num_procedures",
    "num_medications", "lab_abnormalities", "discharge_to_home",
    "admission_burden", "complexity_score", "lab_per_day",
    "high_risk_discharge", "elderly"
]

def load_data(data_path):
    """Load data from CSV file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    return pd.read_csv(data_path)

def engineer_features(df):
    """Engineer 5 clinical features from the 10 raw EHR features."""
    df = df.copy()
    
    # 1. admission_burden: reflect how frequently a patient is admitted weighted by comorbidities
    df["admission_burden"] = df["num_prior_admissions"] * (df["comorbidity_index"] + 1)
    
    # 2. complexity_score: clinical complexity proxy from comorbidities, procedures and diagnoses
    df["complexity_score"] = df["comorbidity_index"] + df["num_procedures"] + df["num_diagnoses"]
    
    # 3. lab_per_day: rate of abnormal lab tests per day of stay
    df["lab_per_day"] = df["lab_abnormalities"] / df["length_of_stay"].clip(lower=1)
    
    # 4. high_risk_discharge: discharge is high risk if not going home, or high prior admissions, or high comorbidities
    df["high_risk_discharge"] = ((df["discharge_to_home"] == 0) | 
                                 (df["comorbidity_index"] >= 5) | 
                                 (df["num_prior_admissions"] >= 3)).astype(int)
    
    # 5. elderly: binary indicator if age >= 65
    df["elderly"] = (df["age"] >= 65).astype(int)
    
    return df

def preprocess(df, fit=True):
    """Clean, engineer features, impute, and scale data."""
    df_engineered = engineer_features(df)
    
    X = df_engineered[FEATURE_COLS]
    y = df_engineered["readmitted"] if "readmitted" in df_engineered.columns else None
    
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    imputer_path = os.path.join(models_dir, "imputer.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    
    if fit:
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        
        X_imputed = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_imputed)
        
        joblib.dump(imputer, imputer_path)
        joblib.dump(scaler, scaler_path)
    else:
        if os.path.exists(imputer_path) and os.path.exists(scaler_path):
            imputer = joblib.load(imputer_path)
            scaler = joblib.load(scaler_path)
        else:
            raise FileNotFoundError("Fitted imputer or scaler not found. Run training (preprocess with fit=True) first.")
        
        X_imputed = imputer.transform(X)
        X_scaled = scaler.transform(X_imputed)
        
    X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLS)
    return X_scaled_df, y

def apply_smote(X_train, y_train):
    """Handle class imbalance by oversampling the minority class using SMOTE."""
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"[INFO] SMOTE Applied. Original shape: {X_train.shape}, Resampled shape: {X_res.shape}")
    return X_res, y_res

def generate_demo_data(data_path="data/demo_data.csv"):
    """Generate a realistic synthetic patient EHR dataset for testing and training."""
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    np.random.seed(42)
    n_samples = 500
    
    age = np.random.randint(18, 90, size=n_samples)
    gender = np.random.randint(0, 2, size=n_samples)
    num_diagnoses = np.random.randint(1, 15, size=n_samples)
    length_of_stay = np.random.randint(1, 21, size=n_samples)
    num_prior_admissions = np.random.randint(0, 6, size=n_samples)
    comorbidity_index = np.random.randint(0, 8, size=n_samples)
    num_procedures = np.random.randint(0, 6, size=n_samples)
    num_medications = np.random.randint(1, 35, size=n_samples)
    lab_abnormalities = np.random.randint(0, 15, size=n_samples)
    discharge_to_home = np.random.choice([1, 0], size=n_samples, p=[0.75, 0.25])
    
    # Calculate readmitted target based on clinical risk factors
    score = (
        -2.5 
        + 0.015 * age 
        + 0.45 * num_prior_admissions 
        + 0.3 * comorbidity_index 
        - 0.6 * discharge_to_home 
        + 0.05 * length_of_stay
        + 0.1 * num_diagnoses
    )
    
    prob = 1 / (1 + np.exp(-score))
    
    # Sort and label the top ~12% highest probabilities to simulate the class imbalance
    threshold = np.percentile(prob, 88)
    readmitted = (prob >= threshold).astype(int)
    
    # Add minor noise (flip labels of 2% random samples)
    noise_mask = np.random.rand(n_samples) < 0.02
    readmitted[noise_mask] = 1 - readmitted[noise_mask]
    
    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "num_diagnoses": num_diagnoses,
        "length_of_stay": length_of_stay,
        "num_prior_admissions": num_prior_admissions,
        "comorbidity_index": comorbidity_index,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "lab_abnormalities": lab_abnormalities,
        "discharge_to_home": discharge_to_home,
        "readmitted": readmitted
    })
    
    df.to_csv(data_path, index=False)
    print(f"[INFO] Synthetic demo dataset generated with {n_samples} records saved to {data_path}")
    print(f"       Class distribution: Not Readmitted={np.sum(readmitted==0)}, Readmitted={np.sum(readmitted==1)}")
