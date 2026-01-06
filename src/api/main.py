import os
import time
import logging
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from ..predict import load_model, predict_one, MODEL_PATH_DEFAULT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("heart_api")

PRED_COUNT = Counter("pred_requests_total", "Total /predict requests")
PRED_LAT = Histogram("pred_latency_seconds", "Latency for /predict endpoint")
PRED_POS = Counter("pred_positive_total", "Total positive predictions")
PRED_NEG = Counter("pred_negative_total", "Total negative predictions")

class HeartFeatures(BaseModel):
    age: int = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: int = Field(..., ge=50, le=250)
    chol: int = Field(..., ge=50, le=700)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: int = Field(..., ge=50, le=250)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=0, le=2)
    ca: int = Field(..., ge=0, le=4)
    thal: int = Field(..., ge=0, le=3)

class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_path: str

app = FastAPI(title="Heart Disease Risk API", version="1.0.0")

_model = None
_model_path = os.getenv("MODEL_PATH", MODEL_PATH_DEFAULT)

@app.on_event("startup")
def startup():
    global _model
    _model = load_model(_model_path)
    logger.info("Model loaded from %s", _model_path)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictResponse)
def predict(payload: HeartFeatures):
    global _model
    PRED_COUNT.inc()
    start = time.time()

    features = payload.model_dump()
    pred, proba = predict_one(_model, features)

    if pred == 1:
        PRED_POS.inc()
    else:
        PRED_NEG.inc()

    latency = time.time() - start
    PRED_LAT.observe(latency)

    logger.info("predict: pred=%s proba=%.4f input=%s", pred, proba, features)

    return PredictResponse(prediction=pred, probability=proba, model_path=_model_path)
