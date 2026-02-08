"""
FastAPI application for churn prediction.
Production-ready API for AWS ECS deployment.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
# Add src to path
sys.path.insert(0, "src")
from inference.inference import (
    ChurnPredictor,
    create_predictor_from_mlflow,
    create_predictor_from_s3,
)






# ============================================================================
# CONFIGURATIONn : get important configuration from environment variables for flexibility in different environments (local, staging, production)
# ============================================================================


class Config:
    """Load configuration from environment variables."""

    APP_NAME = os.getenv("APP_NAME", "Churn Prediction API")
    APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    # Server settings
    HOST = os.getenv("API_HOST", "0.0.0.0")
    PORT = int(os.getenv("API_PORT", "8000"))

    # Model source: 'mlflow' or 's3'
    MODEL_SOURCE = os.getenv("MODEL_SOURCE", "mlflow")

    # MLflow settings (used when MODEL_SOURCE=mlflow)
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    MLFLOW_RUN_ID = os.getenv("MLFLOW_RUN_ID", None)
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Colab_GPU_Training")

    # S3 settings (used when MODEL_SOURCE=s3)
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", None)
    S3_MODEL_NAME = os.getenv("S3_MODEL_NAME", None)
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    S3_MODEL_PREFIX = os.getenv("S3_MODEL_PREFIX", "models/")

    # Model settings
    PREDICTION_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "1000"))

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)









# ============================================================================
# PYDANTIC MODELS : validate data that comes into the API endpoints, ensuring it meets the expected schema and constraints
# ============================================================================


class SubscriptionType(str, Enum):
    BASIC = "Basic"
    STANDARD = "Standard"
    PREMIUM = "Premium"


class ContractLength(str, Enum):
    MONTHLY = "Monthly"
    QUARTERLY = "Quarterly"
    ANNUAL = "Annual"


class Gender(str, Enum):
    MALE = "Male"
    FEMALE = "Female"


class CustomerData(BaseModel):
    """Customer data input schema."""

    customerid: str | None = Field(None, description="Customer ID")
    age: int = Field(..., ge=18, le=100)
    gender: Gender
    tenure: int = Field(..., ge=0, le=100)
    usage_frequency: int = Field(..., ge=0)
    support_calls: int = Field(..., ge=0)
    payment_delay: int = Field(..., ge=0)
    subscription_type: SubscriptionType
    contract_length: ContractLength
    total_spend: float = Field(..., ge=0)
    last_interaction: int = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "customerid": "CUST001",
                "age": 35,
                "gender": "Male",
                "tenure": 24,
                "usage_frequency": 15,
                "support_calls": 3,
                "payment_delay": 5,
                "subscription_type": "Premium",
                "contract_length": "Annual",
                "total_spend": 1250.50,
                "last_interaction": 10,
            }
        }
    }


"""Batch prediction request."""
class BatchPredictionRequest(BaseModel):
   
    customers: list[CustomerData] = Field(..., max_length=1000)



"""Prediction response schema."""
class PredictionResponse(BaseModel):
    
    customerid: str | None
    churn_probability: float
    churn_prediction: int
    risk_level: str

"""Batch prediction response."""
class BatchPredictionResponse(BaseModel):
    
    predictions: list[PredictionResponse]
    total_customers: int
    high_risk_count: int
    churn_rate: float

"""Health check response."""
class HealthResponse(BaseModel):

    status: str
    model_loaded: bool
    model_info: Dict[str, Any] | None = None


# ============================================================================
# GLOBAL PREDICTOR : this class will hold the loaded model and be used across all API requests, ensuring we only load the model once at startup for efficiency
# ============================================================================

predictor: ChurnPredictor | None = None 



# single tone model serving patern - we load the model once at startup and reuse it for all requests, this is more efficient than loading the model on every request, especially for large models that take time to load
# ============================================================================
# LIFESPAN EVENTS
# Startup: Loads model once when server starts (not per request!)
# Shutdown: Cleanup when server stops
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    global predictor # this is the global predictor instance that will be used across all requests, we load the model here once at startup for efficiency

    # Startup
    logger.info("Starting Churn Prediction API...")
    logger.info(f"Model source: {Config.MODEL_SOURCE}")
    
    try:
        if Config.MODEL_SOURCE.lower() == "s3":
            # Load model from S3
            if not Config.S3_MODEL_NAME:
                raise ValueError("S3_MODEL_NAME environment variable is required when using S3")
            
            logger.info(f"Loading model from S3: {Config.S3_MODEL_NAME}")
            predictor = create_predictor_from_s3(
                model_name=Config.S3_MODEL_NAME,
                bucket_name=Config.S3_BUCKET_NAME,
                region=Config.AWS_REGION,
                model_prefix=Config.S3_MODEL_PREFIX,
                threshold=Config.PREDICTION_THRESHOLD,
            )
            logger.info("Model loaded successfully from S3")
        else:
            # Load model from MLflow (default)
            logger.info("Loading model from MLflow")
            predictor = create_predictor_from_mlflow( # this function will load the model from MLflow using the provided run_id and experiment_name, and return a ChurnPredictor instance that we can use for predictions
                run_id=Config.MLFLOW_RUN_ID,
                experiment_name=Config.MLFLOW_EXPERIMENT_NAME,
                tracking_uri=Config.MLFLOW_TRACKING_URI,
                threshold=Config.PREDICTION_THRESHOLD,
            )
            logger.info("Model loaded successfully from MLflow")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("API starting without model")

    yield

    # Shutdown
    logger.info("Shutting down API...")








# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title=Config.APP_NAME,
    version=Config.APP_VERSION,
    description="Production ML API for customer churn prediction",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": str(exc) if Config.DEBUG else "An unexpected error occurred",
        },
    )


# ============================================================================
# API ENDPOINTS
# ============================================================================


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": Config.APP_NAME,
        "version": Config.APP_VERSION,
        "status": "running",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check for AWS ECS."""
    model_loaded = predictor is not None and predictor.model is not None

    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_info=predictor.get_model_info() if model_loaded else None,
    )


@app.get("/ready")
async def readiness_check():
    """Readiness probe."""
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )
    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerData):
    """Predict churn for a single customer."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    try:
        result = predictor.predict(customer.model_dump())
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict churn for multiple customers."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    if len(request.customers) > Config.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size exceeds maximum of {Config.MAX_BATCH_SIZE}",
        )

    try:
        customers_dict = [c.model_dump() for c in request.customers]
        result = predictor.predict_batch(customers_dict)
        return BatchPredictionResponse(**result)
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.get("/model/info")
async def model_info():
    """Get model information."""
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    return predictor.get_model_info()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.DEBUG,
        log_level=Config.LOG_LEVEL.lower(),
    )
