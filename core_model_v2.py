import pandas as pd
import numpy as np
import os
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATA_PATH = "web-page-phishing.csv"
MODEL_SAVE_PATH = "phishing_lgbm_model.joblib"
ARTIFACTS_DIR = "v2_artifacts"

if not os.path.exists(ARTIFACTS_DIR):
    os.makedirs(ARTIFACTS_DIR)

def train_core_model():
    """
    Core Pipeline: Stratified Split -> LightGBM Training with Early Stopping -> Threshold Tuning -> Evaluation
    """
    print("🚀 Starting Phase 1 Model Upgrade...")

    # 1. Load Dataset
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Target column check
    target = 'phishing'
    if target not in df.columns:
        target = df.columns[-1]
    
    X = df.drop(columns=[target])
    y = df[target]

    # 2. Proper Stratified Split (Train 70%, Val 15%, Test 15%)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.176, random_state=42, stratify=y_train_full
    )

    print(f"📊 Dataset split complete:")
    print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 3. LightGBM Pipeline
    # No scaling required for tree-based models
    clf = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        objective='binary',
        metric='auc',
        random_state=42,
        importance_type='gain',
        n_jobs=-1
    )

    print("🛠️ Training LightGBM with Early Stopping...")
    
    # Using the new LightGBM early stopping callback style
    callbacks = [lgb.early_stopping(stopping_rounds=50)]
    
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=callbacks
    )

    # 4. Threshold Tuning based on F1-score
    print("⚖️ Tuning Probability Threshold...")
    y_val_proba = clf.predict_proba(X_val)[:, 1]
    
    thresholds = np.arange(0.1, 0.9, 0.05)
    f1_scores = [f1_score(y_val, (y_val_proba >= t).astype(int)) for t in thresholds]
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"🎯 Optimal Threshold (Val F1): {optimal_threshold:.2f}")

    # 5. Final Evaluation on Test Set
    print("🧪 Evaluating on Test Set...")
    y_test_proba = clf.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_test_pred),
        "Precision": precision_score(y_test, y_test_pred),
        "Recall": recall_score(y_test, y_test_pred),
        "F1-Score": f1_score(y_test, y_test_pred),
        "ROC-AUC": roc_auc_score(y_test, y_test_proba),
        "PR-AUC": average_precision_score(y_test, y_test_proba)
    }

    print("\n📈 Model Benchmarks:")
    import json
    print(json.dumps(metrics, indent=4))

    # 6. Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Phish'], yticklabels=['Legit', 'Phish'])
    plt.title("Phishing Detection Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(os.path.join(ARTIFACTS_DIR, "confusion_matrix_v2.png"))
    plt.close()

    # 7. Feature Importance
    print("🔎 Generating Feature Importance...")
    lgb.plot_importance(clf, max_num_features=15, importance_type='gain', figsize=(10, 8))
    plt.title("LightGBM Feature Importance (Gain)")
    plt.savefig(os.path.join(ARTIFACTS_DIR, "feature_importance_v2.png"))
    plt.close()

    # 8. Save Artifacts
    print("💾 Saving Model & Metadata...")
    joblib.dump(clf, os.path.join(ARTIFACTS_DIR, MODEL_SAVE_PATH))
    metadata = {
        "feature_names": X.columns.tolist(),
        "optimal_threshold": float(optimal_threshold),
        "metrics": metrics
    }
    joblib.dump(metadata, os.path.join(ARTIFACTS_DIR, "model_metadata.joblib"))

    print(f"✅ Phase 1 Upgrade Complete. Artifacts in '{ARTIFACTS_DIR}'")

if __name__ == "__main__":
    train_core_model()
