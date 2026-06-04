import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(__file__))

from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, precision_score,
    recall_score, brier_score_loss, confusion_matrix,
    ConfusionMatrixDisplay, classification_report
)
from sklearn.calibration import calibration_curve
import joblib

from preprocess import load_data, preprocess, generate_demo_data


def compute_ece(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc  = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.mean() * abs(bin_acc - bin_conf)
    return ece


def plot_roc_curve(y_test, y_proba, save_path="models/roc_curve.png"):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"XGBoost (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Patient Readmission Prediction")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[INFO] ROC curve saved to {save_path}")
    plt.close()


def plot_calibration_curve(y_test, y_proba, save_path="models/calibration_curve.png"):
    """Plot reliability diagram (calibration curve)."""
    fraction_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10)
    plt.figure(figsize=(7, 5))
    plt.plot(mean_pred, fraction_pos, "s-", color="steelblue", label="XGBoost (Calibrated)")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Reliability Diagram — Probability Calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[INFO] Calibration curve saved to {save_path}")
    plt.close()


def plot_confusion_matrix(y_test, y_pred, save_path="models/confusion_matrix.png"):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not Readmitted", "Readmitted"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    plt.title("Confusion Matrix — XGBoost")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[INFO] Confusion matrix saved to {save_path}")
    plt.close()


def full_evaluation(model_path="models/xgb_model.pkl",
                    data_path="data/demo_data.csv",
                    threshold=0.4):
    """Run full evaluation suite on the saved model."""
    if not os.path.exists(data_path):
        generate_demo_data(data_path)

    df    = load_data(data_path)
    X, y  = preprocess(df, fit=False)

    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model   = joblib.load(model_path)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    # ── Metrics ───────────────────────────────────────────────────────────────
    auc     = roc_auc_score(y_test, y_proba)
    f1      = f1_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred)
    rec     = recall_score(y_test, y_pred)
    brier   = brier_score_loss(y_test, y_proba)
    ece     = compute_ece(np.array(y_test), y_proba)

    print("\n" + "="*55)
    print("  EVALUATION RESULTS — XGBoost (Calibrated)")
    print("="*55)
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  F1 Score : {f1:.4f}  (threshold = {threshold})")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  Brier    : {brier:.4f}  (lower = better, perfect = 0)")
    print(f"  ECE      : {ece:.4f}  (lower = better, perfect = 0)")
    print("="*55)
    print(f"\n{classification_report(y_test, y_pred, target_names=['Not Readmitted','Readmitted'])}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    plot_roc_curve(y_test, y_proba)
    plot_calibration_curve(y_test, y_proba)
    plot_confusion_matrix(y_test, y_pred)

    return {"auc": auc, "f1": f1, "brier": brier, "ece": ece}


if __name__ == "__main__":
    full_evaluation()
