import pandas as pd
import numpy as np
import os
import joblib
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
)
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Configuration
DATA_PATH = "web-page-phishing.csv"
V3_MODEL_PATH = "v3_artifacts/phishing_lgbm_model_v3.joblib"
ARTIFACTS_DIR = "v4_artifacts"
if not os.path.exists(ARTIFACTS_DIR): os.makedirs(ARTIFACTS_DIR)

def perform_shap_audit(model, X_sample, feature_names):
    logging.info("🔬 Performing SHAP Audit...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # SHAP Summary Plot
    plt.figure(figsize=(10, 8))
    # SHAP in recent versions for binary Lgbm returns a single matrix or list
    try:
        if isinstance(shap_values, list):
            s_vals = shap_values[1]
        else:
            s_vals = shap_values
        shap.summary_plot(s_vals, X_sample, feature_names=feature_names, show=False)
    except Exception as e:
        logging.error(f"SHAP plotting failed: {e}")
    
    plt.title("SHAP Feature Importance (Phase 2 Model)")
    plt.savefig(os.path.join(ARTIFACTS_DIR, "shap_summary_v3.png"))
    plt.close()
    
    # Log diagnostics
    if isinstance(shap_values, list):
        vals = np.abs(shap_values[1]).mean(0)
    else:
        vals = np.abs(shap_values).mean(0)
    feat_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['feature', 'shap_importance'])
    feat_importance = feat_importance.sort_values(by='shap_importance', ascending=False)
    
    logging.info("🔎 Top 10 SHAP Features:")
    logging.info(feat_importance.head(10).to_string(index=False))
    
    lexical_count = sum(1 for f in feat_importance.head(10)['feature'] if any(x in f for x in ['n_', 'url_len']))
    logging.info(f"📊 Lexical Dominance: {lexical_count}/10 features in top ranking are lexical.")
    return feat_importance

def train_diversified_model():
    logging.info("🚀 Phase 3: Signal Diversification & Calibration...")
    
    # 1. Load data and signals (matching Phase 2 structure)
    df = pd.read_csv(DATA_PATH)
    target = 'phishing'
    X = df.drop(columns=[target])
    y = df[target]
    
    # Add the advanced features (matching v3 pipeline for demonstration)
    new_feats = ["domain_age", "domain_entropy", "is_suspicious_tld", "subdomain_depth", "has_A_record", "has_MX_record", "num_nameservers", "dns_ttl", "has_https", "cert_age_days", "days_until_expiry", "is_self_signed", "is_flagged_safe_browsing"]
    for ft in new_feats: 
        if ft not in X.columns: 
            X[ft] = -1 if any(x in ft for x in ["age", "days", "ttl"]) else 0

    feature_names = X.columns.tolist()
    
    # 2. Split
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.176, random_state=42, stratify=y_train_full)

    # 3. SHAP Audit on v3 Model
    if os.path.exists(V3_MODEL_PATH):
        v3_model = joblib.load(V3_MODEL_PATH)
        perform_shap_audit(v3_model, X_val.iloc[:500], feature_names)

    # 4. Diversified Training (v4)
    # Regularization to reduce lexical dominance
    lgbm_v4 = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,           # Reduced from 63 to decrease complexity
        max_depth=6,              # Regularization
        min_child_samples=50,     # Regularization
        lambda_l2=1.0,            # L2 Regularization
        feature_fraction=0.7,     # Forces model to use different signals
        bagging_fraction=0.8,
        bagging_freq=5,
        objective='binary',
        metric='auc',
        random_state=42,
        importance_type='gain',
        n_jobs=-1
    )

    logging.info("🛠️ Training Diversified LightGBM (v4)...")
    lgbm_v4.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=[lgb.early_stopping(50)])

    # 5. Probability Calibration (Isotonic Regression)
    logging.info("⚖️ Applying Probability Calibration (Isotonic)...")
    try:
        calibrator = CalibratedClassifierCV(estimator=lgbm_v4, method='isotonic', cv='prefit')
        calibrator.fit(X_val, y_val)
    except Exception as e:
        logging.error(f"Calibration failed: {e}")
        # Fallback to raw model if calibration fails
        calibrator = lgbm_v4 

    # 6. Evaluation & Calibration Curve
    if hasattr(calibrator, 'predict_proba'):
        y_test_proba_cal = calibrator.predict_proba(X_test)[:, 1]
    else:
        y_test_proba_cal = lgbm_v4.predict_proba(X_test)[:, 1]
        
    y_test_proba_raw = lgbm_v4.predict_proba(X_test)[:, 1]
    
    brier_before = brier_score_loss(y_test, y_test_proba_raw)
    brier_after = brier_score_loss(y_test, y_test_proba_cal) if calibrator != lgbm_v4 else brier_before
    
    logging.info(f"📉 Brier Score: Raw={brier_before:.4f}, Calibrated={brier_after:.4f}")

    # Calibration Plot
    prob_true, prob_pred = calibration_curve(y_test, y_test_proba_cal, n_bins=10)
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.plot(prob_pred, prob_true, marker='.', label='Calibrated model')
    plt.title("Calibration Curve (Isotonic)")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "calibration_curve_v4.png"))
    plt.close()

    # 7. Threshold Tuning
    y_val_proba_cal = calibrator.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.05)
    f1s = [f1_score(y_val, (y_val_proba_cal >= t).astype(int)) for t in thresholds]
    best_t = thresholds[np.argmax(f1s)]
    logging.info(f"🎯 Optimized Calibrated Threshold: {best_t:.2f}")

    # Final Test Set Metrics
    y_test_pred = (y_test_proba_cal >= best_t).astype(int)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_test_pred),
        "Precision": precision_score(y_test, y_test_pred),
        "Recall": recall_score(y_test, y_test_pred),
        "F1": f1_score(y_test, y_test_pred),
        "ROC-AUC": roc_auc_score(y_test, y_test_proba_cal),
        "PR-AUC": average_precision_score(y_test, y_test_proba_cal),
        "Brier_Score": brier_after
    }
    logging.info(f"📊 Final Metrics (v4): {metrics}")

    # 8. Stress Test
    logging.info("🛡️ Running Stress Test...")
    # Simulated top domain test (looking for False Positives)
    X_stress_legit = X_test[y_test == 0].head(100) # Assuming some legit URLs in test set
    fpr_stress = (calibrator.predict_proba(X_stress_legit)[:, 1] >= best_t).mean()
    logging.info(f"🚫 False Positive Rate on Legit Sample: {fpr_stress:.2%}")

    # 9. Save Artifacts
    joblib.dump(calibrator, os.path.join(ARTIFACTS_DIR, "phishing_lgbm_model_v4.joblib"))
    joblib.dump({
        "features": feature_names,
        "threshold": best_t,
        "metrics": metrics,
        "brier_score": brier_after
    }, os.path.join(ARTIFACTS_DIR, "v4_metadata.joblib"))

    logging.info("✅ Phase 3 Optimization Complete.")

if __name__ == "__main__":
    train_diversified_model()
