import os, shutil, json, joblib, datetime, hashlib

class ModelRegistry:
    def __init__(self, base_path="model_registry"):
        self.base_path = base_path
        for folder in ["production", "staging", "archive"]:
            os.makedirs(os.path.join(base_path, folder), exist_ok=True)

    def register_model(self, model_path, metadata, stage="staging"):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        vid = f"v_{ts}"
        vdir = os.path.join(self.base_path, stage, vid)
        os.makedirs(vdir, exist_ok=True)
        tpath = os.path.join(vdir, "model.joblib")
        shutil.copy(model_path, tpath)
        metadata.update({"version_id": vid, "registered_at": ts, "stage": stage})
        with open(os.path.join(vdir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
        return vid

    def load_production_model(self):
        pdir = os.path.join(self.base_path, "production")
        v = sorted(os.listdir(pdir))
        if not v: return None, None
        vdir = os.path.join(pdir, v[-1])
        return joblib.load(os.path.join(vdir, "model.joblib")), json.load(open(os.path.join(vdir, "metadata.json")))

    def promote_to_production(self, vid, source_stage="staging"):
        src = os.path.join(self.base_path, source_stage, vid)
        dst = os.path.join(self.base_path, "production", vid)
        if os.path.exists(src): shutil.copytree(src, dst)
