import pandas as pd
import numpy as np
import joblib, json, os, logging
from monitoring.model_registry import ModelRegistry
from monitoring.logger import ProductionLogger
from monitoring.drift import DriftMonitor
from monitoring.prediction_monitor import PredictionMonitor
from monitoring.dashboard import generate_monitoring_plots

logging.basicConfig(level=logging.INFO)

def run_sim():
    reg = ModelRegistry()
    v4p, v4m = "v4_artifacts/phishing_lgbm_model_v4.joblib", "v4_artifacts/v4_metadata.joblib"
    if not os.path.exists(v4p): return 
    meta = joblib.load(v4m)
    vid = reg.register_model(v4p, meta)
    reg.promote_to_production(vid)
    
    df = pd.read_csv("web-page-phishing.csv")
    for ft in meta["features"]:
        if ft not in df.columns: df[ft] = -1
    base = df[meta["features"]].sample(1000)
    inf = base.copy()
    inf["domain_age"] += 500
    
    drift = DriftMonitor(base).run_drift_check(inf, ["domain_age", "url_length"])
    pred = PredictionMonitor(np.random.rand(100)).analyze_prediction_drift(np.random.rand(100))
    
    logger = ProductionLogger()
    logger.log_inference("test.com", 0.9, 1, vid)
    
    rep = {"drift": drift, "pred": pred, "version": vid}
    json.dump(rep, open("monitoring/daily_report_v4.json", "w"), indent=4)
    generate_monitoring_plots(drift, np.random.rand(100))
    logging.info("Phase 4 Sim Complete.")

if __name__ == "__main__": run_sim()
