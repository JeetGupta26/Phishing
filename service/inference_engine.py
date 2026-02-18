import time, asyncio, numpy as np
from feature_aggregator import FeatureAggregator

class InferenceEngine:
    def __init__(self, m, meta, v):
        self.m, self.meta, self.v = m, meta, v
        self.agg = FeatureAggregator()

    async def predict_url(self, url):
        start = time.time()
        try:
            f = self.agg.extract_all_features(url)
            deg = []
        except Exception as e:
            f = {k: 0 for k in self.meta["features"]}
            deg = [str(e)]
        
        x = np.array([f.get(k, 0) for k in self.meta["features"]]).reshape(1, -1)
        score = float(self.m.predict_proba(x)[:, 1][0])
        dec = "BLOCK" if score >= self.meta["threshold"] else "ALLOW"
        return {"risk_score": score, "decision": dec, "model_version": self.v, "latency_ms": (time.time()-start)*1000, "degradation": deg}
