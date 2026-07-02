from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
import os
import mlflow
import requests
import time
import logging

log = logging.getLogger(__name__)

app = FastAPI(title="VibeX", version="2.0.0")

Instrumentator().instrument(app).expose(app)

os.environ['MLFLOW_TRACKING_URI'] = 'http://mlflow:5000'
mlflow.set_tracking_uri('http://mlflow:5000')

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

FAILURE_PREDICTIONS = Counter('vibex_failure_predictions_total', 'Total FAILURE IMMINENT predictions')
OK_PREDICTIONS = Counter('vibex_ok_predictions_total', 'Total MACHINE OK predictions')
FAILURE_PROBABILITY = Histogram('vibex_failure_probability', 'Failure probability distribution', buckets=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
PREDICTION_LATENCY = Histogram('vibex_prediction_latency_seconds', 'Prediction latency')
MODEL_VERSION = Gauge('vibex_model_version', 'Active model version')

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
        log.error(f"Model load error: {e}")
        return None

def get_llm_explanation(sensor_data: dict, prediction: int, probability: float) -> str:
    try:
        prompt = f"""You are an industrial maintenance expert.
A machine sensor reading shows:
- Air Temperature: {sensor_data['air_temperature']}K
- Process Temperature: {sensor_data['process_temperature']}K
- Rotation Speed: {sensor_data['rotation']} RPM (Revolutions Per Minute)
- Torque: {sensor_data['torque']} Nm (Newton Meters)
- Tool Wear: {sensor_data['tool_wear']} minutes

VibeX AI predicted: {'FAILURE IMMINENT' if prediction == 1 else 'MACHINE OK'}
with {probability*100:.1f}% confidence.

Explain why in 2 sentences. Be specific about which sensors indicate the issue."""

        response = requests.post(
            "http://ollama:11434/api/generate",
            json={"model": "qwen2.5:1.5b", "prompt": prompt, "stream": False},
            timeout=30
        )
        return response.json().get("response", "Explanation unavailable")
    except Exception as e:
        log.error(f"LLM (Large Language Model) error: {e}")
        return "Explanation unavailable"

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: SensorData):
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
    FAILURE_PROBABILITY.observe(float(probability))
    PREDICTION_LATENCY.observe(time.time() - start)
    if prediction == 1:
        FAILURE_PREDICTIONS.inc()
        status = "FAILURE IMMINENT"
    else:
        OK_PREDICTIONS.inc()
        status = "MACHINE OK"
    explanation = get_llm_explanation(data.dict(), int(prediction), float(probability))
    return {
        "prediction": int(prediction),
        "failure_probability": round(float(probability), 4),
        "status": status,
        "explanation": explanation
    }
