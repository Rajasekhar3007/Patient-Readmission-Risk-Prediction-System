import numpy as np
import pandas as pd
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Ensure src directory is in path
sys.path.append(os.path.dirname(__file__))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import xgboost as xgb
import joblib

from preprocess import load_data, preprocess, apply_smote, generate_demo_data


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression with GridSearchCV."""
    print("\n[INFO] Training Logistic Regression...")
    param_grid = {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs"], "max_iter": [1000]}
    lr = LogisticRegression(random_state=42, class_weight="balanced")
    grid = GridSearchCV(lr, param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"  Best params: {grid.best_params_} | CV AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_


def train_random_forest(X_train, y_train):
    """Train Random Forest with GridSearchCV."""
    print("\n[INFO] Training Random Forest...")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [6, 10, None],
        "min_samples_split": [2, 5]
    }
    rf = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)
    grid = GridSearchCV(rf, param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"  Best params: {grid.best_params_} | CV AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_


def train_xgboost(X_train, y_train):
    """Train XGBoost with GridSearchCV and scale_pos_weight for imbalance."""
    print("\n[INFO] Training XGBoost...")
    scale_pos = int((y_train == 0).sum() / (y_train == 1).sum())
    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        scale_pos_weight=scale_pos,
        eval_metric="auc",
        use_label_encoder=False
    )
    grid = GridSearchCV(xgb_model, param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"  Best params: {grid.best_params_} | CV AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_


def calibrate_model(model, X_train, y_train, method="sigmoid"):
    """Apply Platt Scaling (sigmoid) for probability calibration."""
    print(f"\n[INFO] Calibrating model with Platt Scaling ({method})...")
    calibrated = CalibratedClassifierCV(model, method=method, cv=3)
    calibrated.fit(X_train, y_train)
    return calibrated


def evaluate_model(model, X_test, y_test, name="Model", threshold=0.4):
    """Evaluate model with AUC-ROC, F1, and classification report."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  AUC-ROC : {auc:.4f}")
    print(f"  F1 Score: {f1:.4f} (threshold={threshold})")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Not Readmitted', 'Readmitted'])}")
    return {"name": name, "auc": auc, "f1": f1}


def run_stratified_cv(model, X, y, n_splits=5):
    """Run stratified k-fold cross-validation."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    auc_scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    f1_scores  = cross_val_score(model, X, y, cv=skf, scoring="f1", n_jobs=-1)
    print(f"\n[INFO] {n_splits}-Fold CV — AUC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f} | "
          f"F1: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
    return auc_scores, f1_scores


def main():
    # ── 1. Data ──────────────────────────────────────────────────────────────
    if not os.path.exists("data/demo_data.csv"):
        generate_demo_data("data/demo_data.csv")

    df = load_data("data/demo_data.csv")
    X, y = preprocess(df, fit=True)

    # Train/test split (stratified)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Apply SMOTE only on training data
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # ── 2. Train Models ───────────────────────────────────────────────────────
    lr_model  = train_logistic_regression(X_train_res, y_train_res)
    rf_model  = train_random_forest(X_train_res, y_train_res)
    xgb_model = train_xgboost(X_train_res, y_train_res)

    # Calibrate XGBoost
    xgb_calibrated = calibrate_model(xgb_model, X_train_res, y_train_res)

    # ── 3. Evaluate ───────────────────────────────────────────────────────────
    results = []
    results.append(evaluate_model(lr_model,        X_test, y_test, "Logistic Regression"))
    results.append(evaluate_model(rf_model,         X_test, y_test, "Random Forest"))
    results.append(evaluate_model(xgb_model,        X_test, y_test, "XGBoost (Raw)"))
    results.append(evaluate_model(xgb_calibrated,   X_test, y_test, "XGBoost (Calibrated)"))

    # ── 4. Cross-Validation on Best Model ─────────────────────────────────────
    print("\n[INFO] Running 5-Fold Stratified CV on XGBoost...")
    run_stratified_cv(xgb_model, X, y, n_splits=5)

    # ── 5. Save Best Model ────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(xgb_calibrated, "models/xgb_model.pkl")
    print("\n[INFO] Best model (XGBoost Calibrated) saved to models/xgb_model.pkl")

    # Summary table
    print("\n\nModel Comparison Summary")
    print(f"{'Model':<30} {'AUC-ROC':>10} {'F1 Score':>10}")
    print("-" * 52)
    for r in results:
        print(f"{r['name']:<30} {r['auc']:>10.4f} {r['f1']:>10.4f}")


if __name__ == "__main__":
    main()
