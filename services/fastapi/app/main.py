from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="PGuard", version="1.0.0")

Instrumentator().instrument(app).expose(app)

class SensorData(BaseModel):
    temperature: float
    rotation: float
    torque: float
    tool_wear: float

@app.get("/")
def root():
    return {"service": "PGuard", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: SensorData):
    try:
        model = mlflow.sklearn.load_model("models:/PGuard/Production")
        features = np.array([[
            data.temperature,
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
    except Exception as e:
        return {
            "error": str(e),
            "status": "MODEL NOT LOADED YET"
        }
