import numpy as np
from scipy.stats import ks_2samp

class DriftMonitor:
    def __init__(self, baseline_df):
        self.baseline_df = baseline_df

    def calculate_psi(self, expected, actual, buckets=10):
        def get_counts(arr, bins):
            c, _ = np.histogram(arr, bins=bins)
            return c / len(arr)
        if len(expected) < buckets or len(actual) < buckets: return 0.0
        bins = np.histogram_bin_edges(expected, bins=buckets)
        ep = np.clip(get_counts(expected, bins), 0.0001, None)
        ap = np.clip(get_counts(actual, bins), 0.0001, None)
        return float(np.sum((ap - ep) * np.log(ap / ep)))

    def run_drift_check(self, df, feats):
        res = {"feature_drifts": {}}
        alert = False
        for f in feats:
            if f not in df.columns: continue
            e, a = self.baseline_df[f].dropna(), df[f].dropna()
            psi = self.calculate_psi(e, a)
            ks_stat, p = ks_2samp(e, a)
            status = "NONE"
            if psi > 0.25: status = "CRITICAL"; alert = True
            elif psi > 0.10: status = "MODERATE"
            res["feature_drifts"][f] = {"psi": float(psi), "status": status}
        res["alert_triggered"] = bool(alert)
        return res
