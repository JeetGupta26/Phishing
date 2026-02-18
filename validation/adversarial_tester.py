import numpy as np
from feature_aggregator import FeatureAggregator

class AdversarialTester:
    def __init__(self, model, metadata):
        self.m, self.meta = model, metadata
        self.agg = FeatureAggregator()

    def run_tests(self):
        urls = ["http://paypaI.com", "http://xn--80ak6aa92e.com", "http://127.0.0.1@microsoft.com"]
        res = []
        for u in urls:
            f = self.agg.extract_all_features(u)
            x = np.array([f.get(k, 0) for k in self.meta["features"]]).reshape(1, -1)
            p = float(self.m.predict_proba(x)[:, 1][0])
            res.append({"url": u, "score": p, "dec": "BLOCK" if p >= self.meta["threshold"] else "ALLOW"})
        return {"results": res}
