import pandas as pd
import numpy as np
import os
import joblib
import lightgbm as lgb
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from feature_aggregator import extract_all_features

logging.basicConfig(level=logging.INFO, format='%(message)s')

DATA_PATH = "web-page-phishing.csv"
ARTIFACTS_DIR = "v3_artifacts"
if not os.path.exists(ARTIFACTS_DIR): os.makedirs(ARTIFACTS_DIR)

def train_v3():
    logging.info("🚀 Phase 2: Multi-Signal Training...")
    df = pd.read_csv(DATA_PATH)
    target = 'phishing'
    X = df.drop(columns=[target])
    y = df[target]
    
    new_feats = ["domain_age", "domain_entropy", "is_suspicious_tld", "subdomain_depth", "has_A_record", "has_MX_record", "num_nameservers", "dns_ttl", "has_https", "cert_age_days", "days_until_expiry", "is_self_signed", "is_flagged_safe_browsing"]
    for ft in new_feats: 
        if ft not in X.columns: 
            X[ft] = -1 if any(x in ft for x in ["age", "days", "ttl"]) else 0
            
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.176, random_state=42, stratify=y_train_full)

    clf = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=63, objective='binary', metric='auc', random_state=42, importance_type='gain', n_jobs=-1)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=[lgb.early_stopping(50)])

    y_val_proba = clf.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.05)
    f1s = [f1_score(y_val, (y_val_proba >= t).astype(int)) for t in thresholds]
    best_t = thresholds[np.argmax(f1s)]
    
    y_test_proba = clf.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= best_t).astype(int)
    
    metrics = {"Accuracy": accuracy_score(y_test, y_test_pred), "Precision": precision_score(y_test, y_test_pred), "Recall": recall_score(y_test, y_test_pred), "F1": f1_score(y_test, y_test_pred), "ROC-AUC": roc_auc_score(y_test, y_test_proba), "PR-AUC": average_precision_score(y_test, y_test_proba)}
    logging.info(f"📈 Metrics: {metrics}")
    
    joblib.dump(clf, os.path.join(ARTIFACTS_DIR, "phishing_lgbm_model_v3.joblib"))
    joblib.dump({"features": X.columns.tolist(), "threshold": best_t, "metrics": metrics}, os.path.join(ARTIFACTS_DIR, "v3_metadata.joblib"))
    
    lgb.plot_importance(clf, max_num_features=15, importance_type='gain', figsize=(10, 8))
    plt.title("Importance v3")
    plt.savefig(os.path.join(ARTIFACTS_DIR, "feature_importance_v3.png"))
    plt.close()
    logging.info("✅ Phase 2 Upgrade Complete.")

if __name__ == "__main__": train_v3()
