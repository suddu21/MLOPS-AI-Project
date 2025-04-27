import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from prometheus_client import Counter, Gauge, Histogram, start_http_server
import os
import logging
from datetime import datetime

# Initialize Prometheus metrics
REQUEST_COUNT = Counter('request_count', 'Total request count')
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency in seconds')
CPU_USAGE = Gauge('cpu_usage', 'CPU usage percentage')
MEMORY_USAGE = Gauge('memory_usage', 'Memory usage in MB')

# Initialize FastAPI app
app = FastAPI()

# Load pre-trained model (placeholder, to be replaced with MLflow model)
try:
    model = joblib.load("models/fraud_model.joblib")
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Model not found")

# Define input data model
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# Feedback storage (in-memory for simplicity, replace with database in production)
feedback_log = []

# Mock system metrics (replace with actual monitoring in production)
def update_metrics():
    CPU_USAGE.set(50.0)  # Placeholder value
    MEMORY_USAGE.set(1024.0)  # Placeholder value

@app.post("/predict")
async def predict(transaction: Transaction):
    REQUEST_COUNT.inc()
    start_time = datetime.now()

    # Convert input to DataFrame
    data = pd.DataFrame([transaction.dict().values()], columns=transaction.dict().keys())
    
    # Predict
    prediction = model.predict(data)
    probability = model.predict_proba(data)[:, 1][0]

    # Log latency
    latency = (datetime.now() - start_time).total_seconds()
    REQUEST_LATENCY.observe(latency)

    # Update system metrics
    update_metrics()

    # Store feedback (e.g., manual validation by user)
    feedback_log.append({"transaction": transaction.dict(), "prediction": int(prediction[0]), "probability": float(probability)})

    # Check feedback loop (simplified: retrain if 10% of predictions are incorrect)
    if len(feedback_log) > 10:
        correct_count = sum(1 for f in feedback_log[-10:] if f.get("manual_label", f["prediction"]) == f["prediction"])
        if correct_count / 10 < 0.9:
            trigger_retraining()

    return {"prediction": int(prediction[0]), "probability": float(probability)}

def trigger_retraining():
    # Placeholder for retraining logic (e.g., call Airflow DAG or MLflow run)
    logging.info("Model retraining triggered due to feedback loop")
    # In production, this would invoke an Airflow DAG or MLflow pipeline

if __name__ == "__main__":
    start_http_server(8000)  # Prometheus metrics endpoint
    uvicorn.run(app, host="0.0.0.0", port=8000)