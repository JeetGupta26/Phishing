import pandas as pd
import hashlib, os, logging

class RetrainingDatasetBuilder:
    def __init__(self, logs_path="monitoring/inference.log", baseline_path="web-page-phishing.csv"):
        self.logs_path, self.baseline_path = logs_path, baseline_path

    def build_dataset(self, feature_names, output_path="retraining/dataset_v5_sim.csv"):
        df_base = pd.read_csv(self.baseline_path)
        
        # Synthesize multi-signal features if missing (as done in previous phases)
        for ft in feature_names:
            if ft not in df_base.columns:
                df_base[ft] = -1 if any(x in ft for x in ["age", "days", "ttl"]) else 0
        
        new_samples = df_base.sample(500).copy()
        df_retrain = pd.concat([df_base, new_samples]).drop_duplicates()
        df_retrain.to_csv(output_path, index=False)
        d_hash = hashlib.sha256(df_retrain.to_csv().encode()).hexdigest()
        logging.info(f"Dataset Built: {len(df_retrain)} rows. Hash: {d_hash[:10]}")
        return output_path, d_hash
