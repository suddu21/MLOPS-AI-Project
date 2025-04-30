import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
import psutil
import pickle
import logging
import os

# Initialize FastAPI app
app = FastAPI(title="Fraud Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus WSGI middleware to route /metrics requests
prometheus_app = make_asgi_app()
app.mount("/metrics", prometheus_app)

# Prometheus metrics
REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["endpoint", "client_ip"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])
CPU_USAGE = Gauge("system_cpu_usage_percent", "CPU usage percentage")
MEMORY_USAGE = Gauge("system_memory_usage_percent", "Memory usage percentage")
DISK_IO_READ = Counter("system_disk_io_read_bytes", "Disk read bytes")
DISK_IO_WRITE = Counter("system_disk_io_write_bytes", "Disk write bytes")
NETWORK_IO_SENT = Counter("system_network_io_sent_bytes", "Network sent bytes")
NETWORK_IO_RECEIVED = Counter("system_network_io_received_bytes", "Network received bytes")
FILE_HANDLES = Gauge("system_file_handles", "Number of open file handles")

# Load the model, scaler, and feature names
MODEL_PATH = "/app/models/fraud_model.pkl"
try:
    with open(MODEL_PATH, 'rb') as f:
        saved_data = pickle.load(f)
    model = saved_data['model']
    scaler = saved_data['scaler']
    feature_names = saved_data['feature_names']
    logging.info("Model, scaler, and feature names loaded successfully")
    logging.info(f"Expected feature names (order matters): {feature_names}")
except Exception as e:
    logging.error(f"Failed to load model: {str(e)}")
    raise HTTPException(status_code=500, detail="Model loading failed")

# Update system metrics
def update_system_metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=1))
    MEMORY_USAGE.set(psutil.virtual_memory().percent)
    disk_io = psutil.disk_io_counters()
    DISK_IO_READ.inc(disk_io.read_bytes)
    DISK_IO_WRITE.inc(disk_io.write_bytes)
    net_io = psutil.net_io_counters()
    NETWORK_IO_SENT.inc(net_io.bytes_sent)
    NETWORK_IO_RECEIVED.inc(net_io.bytes_recv)
    FILE_HANDLES.set(len(psutil.Process().open_files()))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    REQUEST_COUNT.labels(endpoint="/predict", client_ip="unknown").inc()
    with REQUEST_LATENCY.labels(endpoint="/predict").time():
        update_system_metrics()
        try:
            # Validate file type
            if not file.filename.endswith('.csv'):
                raise HTTPException(status_code=400, detail="File must be a CSV")

            # Read the file content
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")

            # Log the raw content for debugging
            logging.info(f"Raw file content: {content.decode('utf-8', errors='ignore')}")

            # Parse the CSV
            try:
                df = pd.read_csv(pd.io.common.StringIO(content.decode('utf-8')))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

            # Log the parsed DataFrame shape and columns
            logging.info(f"Parsed DataFrame shape: {df.shape}")
            logging.info(f"Parsed DataFrame columns: {list(df.columns)}")

            # Validate required columns
            required_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
            if not all(col in df.columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in df.columns]  # Fixed logic
                raise HTTPException(status_code=400, detail=f"CSV missing required columns. Missing: {missing_cols}, Expected: {required_columns}")

            # Reorder columns to match training order
            X = df[feature_names]
            logging.info(f"Reordered DataFrame columns: {list(X.columns)}")

            # Preprocess and predict
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled).tolist()
            probabilities = model.predict_proba(X_scaled)[:, 1].tolist()

            # Prepare response
            result = []
            for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
                result.append({
                    "row": idx + 1,
                    "data": df.iloc[idx].to_dict(),
                    "prediction": "Fraud" if pred == 1 else "Not Fraud",
                    "fraud_probability": prob
                })

            return JSONResponse(content={"predictions": result})

        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    REQUEST_COUNT.labels(endpoint="/health", client_ip="unknown").inc()
    with REQUEST_LATENCY.labels(endpoint="/health").time():
        update_system_metrics()
        return {"status": "healthy"}