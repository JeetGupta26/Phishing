import numpy as np

class PredictionMonitor:
    def __init__(self, baseline_scores):
        self.baseline_scores = baseline_scores

    def analyze_prediction_drift(self, recent):
        if len(recent) == 0: return {}
        m_base, m_inf = np.mean(self.baseline_scores), np.mean(recent)
        return {
            "mean_shift": float(m_inf - m_base),
            "anomaly_detected": bool(abs(m_inf - m_base) > 0.2),
            "dist": {"low": float((recent < 0.3).mean()), "high": float((recent > 0.7).mean())}
        }
