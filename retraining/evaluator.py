from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score
import numpy as np

def eval_perf(m, X, y, t):
    p = m.predict_proba(X)[:, 1]
    return {"roc_auc": float(roc_auc_score(y, p)), "brier_score": float(brier_score_loss(y, p))}

class ChampionChallengerEvaluator:
    def compare(self, champ, chal, X_t, y_t, ct, chalt):
        cm = eval_perf(champ, X_t, y_t, ct)
        chm = eval_perf(chal, X_t, y_t, chalt)
        promote = (chm["roc_auc"] >= cm["roc_auc"] * 0.99)
        return promote, {"champion": cm, "challenger": chm}
