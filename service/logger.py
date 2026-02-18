import json, sys, datetime, hashlib
class ProductionLogger:
    @staticmethod
    def log(url, res):
        log = {"ts": str(datetime.datetime.now()), "url": hashlib.sha256(url.encode()).hexdigest(), "score": res["risk_score"], "dec": res["decision"], "lat": res["latency_ms"]}
        sys.stdout.write(json.dumps(log) + "\n")
