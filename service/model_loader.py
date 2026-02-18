import joblib, json, os, logging
from service.config import settings
class ModelLoader:
    @staticmethod
    def load_prod():
        p = os.path.join(settings.REGISTRY_PATH, settings.PRODUCTION_STAGE)
        v = sorted(os.listdir(p))[-1]
        vdir = os.path.join(p, v)
        m = joblib.load(os.path.join(vdir, "model.joblib"))
        meta = json.load(open(os.path.join(vdir, "metadata.json")))
        return m, meta, v
