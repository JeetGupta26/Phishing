import logging, pandas as pd, joblib, os
from retraining.dataset_builder import RetrainingDatasetBuilder
from retraining.trigger_logic import check_retraining_trigger
from retraining.trainer import ChallengerTrainer
from retraining.evaluator import ChampionChallengerEvaluator
from retraining.retraining_report import generate_retraining_report
from monitoring.model_registry import ModelRegistry
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

def run_sim():
    logging.info("🚀 Phase 5 Sim...")
    t, reas = check_retraining_trigger()
    if not t: reas = "Sim Forced"
    
    reg = ModelRegistry()
    champ, cmeta = reg.load_production_model()
    if not champ:
        logging.error("No Champ model in production registry.")
        return

    bld = RetrainingDatasetBuilder()
    dpath, dhash = bld.build_dataset(cmeta["features"])
    
    trn = ChallengerTrainer(cmeta["features"])
    chal_path, chal_t = trn.train(dpath)
    chal = joblib.load(chal_path)

    df = pd.read_csv(dpath)
    X, y = df[cmeta["features"]], df['phishing']
    _, Xt, _, yt = train_test_split(X, y, test_size=0.2, random_state=123)

    evl = ChampionChallengerEvaluator()
    promote, comp = evl.compare(champ, chal, Xt, yt, cmeta["threshold"], chal_t)

    dec = "REJECTED"
    vid = "v_failed"
    if promote:
        logging.info("🏆 PROMOTED")
        meta = {"features": cmeta["features"], "threshold": chal_t, "metrics": comp["challenger"]}
        vid = reg.register_model(chal_path, meta)
        reg.promote_to_production(vid)
        dec = "PROMOTED"
    
    generate_retraining_report(vid, dec, comp, reas)
    logging.info("✅ Phase 5 Complete.")

if __name__ == "__main__": run_sim()
