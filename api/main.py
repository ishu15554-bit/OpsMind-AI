
from pathlib import Path

# FastAPI components
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# MLflow Registry
import mlflow
from mlflow.tracking import MlflowClient

# RAG answer generation
from genai.rag.answer_generator import generate_answer

# Database functions
from src.database.db import (
    save_prediction,
    get_prediction_history
)

# ML prediction functions
from src.ml.predict import (
    predict_ticket,
    predict_hdfs_anomaly,
    predict_openstack_anomaly
)


# ==========================================
# MLFLOW CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]
TRACKING_DB = BASE_DIR / "mlops" / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{TRACKING_DB.as_posix()}"
)

mlflow_client = MlflowClient()

REGISTERED_MODELS = {
    "Ticket Classifier": "OpsMind-Ticket-Classifier",
    "OpenStack Anomaly Detection": "OpsMind-OpenStack-Anomaly",
    "HDFS Anomaly Detection": "OpsMind-HDFS-Anomaly"
}


# ==========================================
# INITIALIZE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="OpsMind AI",
    description="Enterprise AIOps Incident Intelligence API",
    version="1.1.0"
)


# ==========================================
# REQUEST VALIDATION
# ==========================================

class TicketRequest(BaseModel):
    ticket_text: str = Field(min_length=1, max_length=10000)


class LogRequest(BaseModel):
    log_text: str = Field(min_length=1, max_length=50000)


# ==========================================
# HOME / HEALTH ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Welcome to OpsMind AI API",
        "status": "Running"
    }


# ==========================================
# ITSM TICKET CLASSIFICATION
# ==========================================

@app.post("/predict/ticket")
def ticket_prediction(request: TicketRequest):

    try:
        result = predict_ticket(request.ticket_text)

        save_prediction(
            "ITSM Ticket Classifier",
            request.ticket_text,
            result
        )

        return {"prediction": str(result)}

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Ticket prediction failed."
        ) from error


# ==========================================
# HDFS ANOMALY DETECTION
# ==========================================

@app.post("/predict/hdfs")
def hdfs_prediction(request: LogRequest):

    try:
        result = predict_hdfs_anomaly(request.log_text)

        save_prediction(
            "HDFS Anomaly Detector",
            request.log_text,
            result
        )

        return {"prediction": result}

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="HDFS prediction failed."
        ) from error


# ==========================================
# OPENSTACK ANOMALY DETECTION
# ==========================================

@app.post("/predict/openstack")
def openstack_prediction(request: LogRequest):

    try:
        result = predict_openstack_anomaly(request.log_text)

        save_prediction(
            "OpenStack Anomaly Detector",
            request.log_text,
            result
        )

        return {"prediction": result}

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="OpenStack prediction failed."
        ) from error


# ==========================================
# RAG TROUBLESHOOTING ASSISTANT
# ==========================================

@app.post("/rag/ask")
def rag_question(request: LogRequest):

    try:
        result = generate_answer(request.log_text)
        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="RAG answer generation failed."
        ) from error


# ==========================================
# DATABASE PREDICTION HISTORY
# ==========================================

@app.get("/predictions/history")
def prediction_history(limit: int = 100):

    try:

        if limit < 1 or limit > 1000:
            raise HTTPException(
                status_code=400,
                detail="Limit must be between 1 and 1000."
            )

        records = get_prediction_history(limit=limit)

        return {
            "status": "success",
            "total_records": len(records),
            "history": records
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Prediction history retrieval failed."
        ) from error


# ==========================================
# MLFLOW MODEL REGISTRY
# ==========================================

@app.get("/models/registry")
def model_registry():

    try:
        registry_results = []

        for display_name, registered_name in REGISTERED_MODELS.items():

            versions = mlflow_client.search_model_versions(
                f"name='{registered_name}'"
            )

            model_versions = []

            for version in versions:

                model_versions.append({
                    "version": version.version,
                    "status": version.status,
                    "run_id": version.run_id,
                    "source": version.source
                })

            registry_results.append({
                "model_name": display_name,
                "registered_name": registered_name,
                "versions": model_versions
            })

        return {
            "status": "success",
            "total_models": len(registry_results),
            "models": registry_results
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Model registry retrieval failed."
        ) from error
