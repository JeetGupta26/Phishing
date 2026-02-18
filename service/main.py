import os, sys
from fastapi import FastAPI, HTTPException
import uvicorn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service.model_loader import ModelLoader
from service.inference_engine import InferenceEngine
from service.schema import PredictRequest, PredictResponse, HealthResponse
from service.logger import ProductionLogger

app = FastAPI()
eng = None
v = None

@app.on_event("startup")
async def startup():
    global eng, v
    m, meta, v = ModelLoader.load_prod()
    eng = InferenceEngine(m, meta, v)

@app.post("/predict", response_model=PredictResponse)
async def predict(r: PredictRequest):
    res = await eng.predict_url(r.url)
    ProductionLogger.log(r.url, res)
    return PredictResponse(**res)

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", ready=(eng is not None))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
