
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict

app = FastAPI(
    title="Blind Assistant Inference Service",
    description="Python service for handling AI inference requests",
    version="1.0.0"
)


# Request structure
class InferenceRequest(BaseModel):
    task: str
    input: Dict[str, Any]


# Health Check Endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "inference-service"
    }


# Fake Inference Endpoint
@app.post("/inference")
def fake_inference(request: InferenceRequest):
    return {
        "status": "success",
        "task": request.task,
        "result": {
            "message": "This is a mock inference response",
            "prediction": "Sample AI result"
        }
    }