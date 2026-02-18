import joblib, os, logging, pandas as pd, numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score

class ChallengerTrainer:
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def train(self, dataset_path, artifacts_dir="retraining/temp_artifacts"):
        os.makedirs(artifacts_dir, exist_ok=True)
        df = pd.read_csv(dataset_path)
        X, y = df[self.feature_names], df['phishing']
        X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.03, num_leaves=31, max_depth=6, lambda_l2=1.0, feature_fraction=0.7, objective='binary', metric='auc', random_state=42)
        clf.fit(X_t, y_t, eval_set=[(X_v, y_v)], callbacks=[lgb.early_stopping(50)])
        
        try:
            cal = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
            cal.fit(X_v, y_v)
            model_to_save = cal
        except Exception as e:
            logging.error(f"Calibration failed in retraining: {e}")
            model_to_save = clf
        
        if hasattr(model_to_save, 'predict_proba'):
            probs = model_to_save.predict_proba(X_v)[:, 1]
        else:
            probs = model_to_save.predict(X_v) # Fallback if needed
            
        ts = np.arange(0.1, 0.9, 0.05)
        best_t = ts[np.argmax([f1_score(y_v, (probs >= t).astype(int)) for t in ts])]
        m_path = os.path.join(artifacts_dir, "challenger_model.joblib")
        joblib.dump(model_to_save, m_path)
        return m_path, best_t
