from pydantic import BaseModel
from typing import Optional, List
class PredictRequest(BaseModel):
    url: str
class PredictResponse(BaseModel):
    risk_score: float
    decision: str
    model_version: str
    latency_ms: float
    degradation: List[str]
class HealthResponse(BaseModel):
    status: str
    ready: bool
