import json, logging, datetime, hashlib, os

class ProductionLogger:
    def __init__(self, log_file="monitoring/inference.log"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.logger = logging.getLogger("ProdInference")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            self.logger.addHandler(logging.FileHandler(log_file))

    def log_inference(self, url, score, decision, model_version, features=None):
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model_version": model_version,
            "url_hash": hashlib.sha256(url.encode()).hexdigest(),
            "risk_score": float(score),
            "decision": int(decision),
            "features": features or {}
        }
        self.logger.info(json.dumps(entry))
