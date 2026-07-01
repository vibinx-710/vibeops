from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
import os
import mlflow

app = FastAPI(title="VibeX", version="1.0.0")

Instrumentator().instrument(app).expose(app)

os.environ['MLFLOW_TRACKING_URI'] = 'http://mlflow:5000'
mlflow.set_tracking_uri('http://mlflow:5000')

# ─── Custom Prometheus Metrics ────────────────────
FAILURE_PREDICTIONS = Counter(
    'vibex_failure_predictions_total',
    'Total number of FAILURE IMMINENT predictions'
)

OK_PREDICTIONS = Counter(
    'vibex_ok_predictions_total', 
    'Total number of MACHINE OK predictions'
)

FAILURE_PROBABILITY = Histogram(
    'vibex_failure_probability',
    'Distribution of failure probability scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

PREDICTION_LATENCY = Histogram(
    'vibex_prediction_latency_seconds',
    'Time taken to make a prediction'
)

MODEL_VERSION = Gauge(
    'vibex_model_version',
    'Currently serving model version'
)

class SensorData(BaseModel):
    air_temperature: float
    process_temperature: float
    rotation: float
    torque: float
    tool_wear: float

def load_model():
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions("Vibex")
        if versions:
            MODEL_VERSION.set(int(versions[0].version))
        model = mlflow.sklearn.load_model("models:/Vibex/latest")
        return model
    except Exception as e:
        print(f"Model load error: {e}")
        return None

@app.get("/")
def root():
    return {"service": "VibeX", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: SensorData):
    import time
    start = time.time()
    
    model = load_model()
    if model is None:
        return {"error": "Model not available", "status": "MODEL NOT LOADED"}
    
    features = np.array([[
        data.air_temperature,
        data.process_temperature,
        data.rotation,
        data.torque,
        data.tool_wear
    ]])
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    # Record custom metrics
    FAILURE_PROBABILITY.observe(float(probability))
    PREDICTION_LATENCY.observe(time.time() - start)
    
    if prediction == 1:
        FAILURE_PREDICTIONS.inc()
        status = "FAILURE IMMINENT"
    else:
        OK_PREDICTIONS.inc()
        status = "MACHINE OK"
    
    return {
        "prediction": int(prediction),
        "failure_probability": round(float(probability), 4),
        "status": status
    }
