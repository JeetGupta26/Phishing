import time, asyncio, numpy as np
from feature_aggregator import FeatureAggregator

class LoadTester:
    def __init__(self, m, meta):
        self.m, self.meta = m, meta
        self.agg = FeatureAggregator()

    async def simulate_load(self, num=30, con=3):
        lats = []
        for i in range(0, num, con):
            start = time.time()
            # Batch simulate
            for _ in range(con): self.agg.extract_all_features("http://test.com")
            lats.append((time.time() - start) * 1000 / con)
        return {"p95_latency_ms": float(np.percentile(lats, 95))}
