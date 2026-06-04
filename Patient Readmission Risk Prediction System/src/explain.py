import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import joblib
import os
import sys

sys.path.append(os.path.dirname(__file__))

from preprocess import load_data, preprocess, generate_demo_data

FEATURE_COLS = [
    "age", "gender", "num_diagnoses", "length_of_stay",
    "num_prior_admissions", "comorbidity_index", "num_procedures",
    "num_medications", "lab_abnormalities", "discharge_to_home",
    "admission_burden", "complexity_score", "lab_per_day",
    "high_risk_discharge", "elderly"
]


def load_model_and_data(model_path="models/xgb_model.pkl",
                        data_path="data/demo_data.csv"):
    if not os.path.exists(data_path):
        generate_demo_data(data_path)
    df   = load_data(data_path)
    X, y = preprocess(df, fit=False)

    from sklearn.model_selection import train_test_split
    _, X_test, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = joblib.load(model_path)
    # Extract base XGBoost estimator from calibrated wrapper for SHAP
    try:
        base_model = model.calibrated_classifiers_[0].estimator
    except AttributeError:
        base_model = model

    return base_model, X_test


def global_shap_summary(model, X_test, save_path="models/shap_summary.png"):
    """Generate SHAP beeswarm summary plot (global feature importance)."""
    print("[INFO] Computing SHAP values for global summary...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=FEATURE_COLS,
                      show=False, plot_size=(10, 6))
    plt.title("SHAP Global Feature Importance", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] SHAP summary plot saved to {save_path}")
    plt.close()

    # Top 5 features by mean |SHAP|
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:5]
    print("\nTop 5 Predictive Features (mean |SHAP|):")
    for rank, i in enumerate(top_idx, 1):
        print(f"  {rank}. {FEATURE_COLS[i]:<30}  mean |SHAP| = {mean_abs[i]:.4f}")

    return shap_values


def shap_bar_plot(model, X_test, save_path="models/shap_bar.png"):
    """Generate SHAP bar plot of global feature importance."""
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=FEATURE_COLS,
                      plot_type="bar", show=False, plot_size=(9, 5))
    plt.title("SHAP Feature Importance (Bar)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] SHAP bar plot saved to {save_path}")
    plt.close()


def local_explanation(model, X_test, patient_idx=0):
    """
    Generate a local SHAP explanation for a single patient.
    Returns top contributing features and their SHAP values.
    """
    print(f"\n[INFO] Local explanation for patient index {patient_idx}...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    patient_shap = shap_values[patient_idx]
    patient_feat = X_test.iloc[patient_idx]

    explanation = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "value":      patient_feat.values,
        "shap_value": patient_shap
    }).sort_values("shap_value", ascending=False, key=abs)

    print(f"\n  Patient {patient_idx} — Top Contributing Features:")
    print(f"  {'Feature':<30} {'Value':>10} {'SHAP':>10}")
    print("  " + "-" * 52)
    for _, row in explanation.head(5).iterrows():
        direction = "+ risk" if row["shap_value"] > 0 else "- risk"
        print(f"  {row['feature']:<30} {row['value']:>10.3f} {row['shap_value']:>10.4f}  {direction}")

    return explanation


def run_full_shap_analysis():
    model, X_test = load_model_and_data()
    os.makedirs("models", exist_ok=True)

    global_shap_summary(model, X_test)
    shap_bar_plot(model, X_test)
    local_explanation(model, X_test, patient_idx=0)
    local_explanation(model, X_test, patient_idx=1)
    print("\n[INFO] SHAP analysis complete.")


if __name__ == "__main__":
    run_full_shap_analysis()
