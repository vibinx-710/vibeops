from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np
from prometheus_fastapi_instrumentator import Instrumentator
import os
import mlflow

app = FastAPI(title="VibeX", version="1.0.0")

Instrumentator().instrument(app).expose(app)

os.environ['MLFLOW_TRACKING_URI'] = 'http://mlflow:5000'
mlflow.set_tracking_uri('http://mlflow:5000')

class SensorData(BaseModel):
    air_temperature: float
    process_temperature: float
    rotation: float
    torque: float
    tool_wear: float

def load_model():
    try:
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
    
    return {
        "prediction": int(prediction),
        "failure_probability": round(float(probability), 4),
        "status": "FAILURE IMMINENT" if prediction == 1 else "MACHINE OK"
    }
