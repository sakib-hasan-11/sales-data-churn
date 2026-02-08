"""
Page 5: API Deployment Documentation
Covers FastAPI, Docker, and AWS ECS deployment
"""

import streamlit as st

st.set_page_config(page_title="API Deployment", page_icon="🌐", layout="wide")

# Custom CSS
st.markdown(
    """
<style>
    .file-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1976D2;
        margin-top: 2rem;
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #1976D2;
    }
    .function-box {
        background-color: #E8F5E9;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #43A047;
    }
    .endpoint-box {
        background-color: #FFF3E0;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #FB8C00;
    }
    .function-name {
        font-family: 'Courier New', monospace;
        font-weight: bold;
        color: #2E7D32;
        font-size: 1.2rem;
    }
    .endpoint-name {
        font-family: 'Courier New', monospace;
        font-weight: bold;
        color: #E65100;
        font-size: 1.2rem;
    }
    .param {
        background-color: #FFF9C4;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
    }
    .return {
        background-color: #C8E6C9;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
    }
    .code-method {
        background-color: #FFEBEE;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
        color: #C62828;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.title("🌐 API Deployment")
st.markdown("### Production FastAPI with Docker & AWS ECS")
st.markdown("---")

# Overview
st.markdown("""
## 📋 Overview

The API deployment system provides:
- 🚀 **REST API** for real-time predictions
- 🐳 **Docker containers** for consistency
- ☁️ **AWS ECS** ready for cloud deployment
- 🔍 **Health checks** for monitoring
- 📊 **Batch processing** for efficiency

**Stack**:
- FastAPI 0.115.0 (async Python web framework)
- Uvicorn (ASGI server)
- Docker (containerization)
- AWS ECS/Fargate (orchestration)
""")

st.markdown("---")

# ============================================================================
# FILE: main.py
# ============================================================================

st.markdown('<div class="file-header">📄 File: main.py</div>', unsafe_allow_html=True)

st.markdown("""
**Purpose**: FastAPI application for churn prediction API

**Location**: `main.py` (project root)

**Key Components**:
1. `Config` - Environment configuration
2. Pydantic models - Request/response validation
3. API endpoints - Prediction, health, info
4. Lifespan management - Model loading/cleanup
""")

st.markdown("---")

# Config Class
st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown('<div class="function-name">⚙️ Config Class</div>', unsafe_allow_html=True)
st.markdown(
    """
**Purpose**: Environment configuration from .env file

**Configuration Variables**:
```python
MLFLOW_TRACKING_URI = "./mlruns"          # MLflow location
MLFLOW_EXPERIMENT_NAME = "Colab_GPU_Training"  # Experiment name
MODEL_OPTIMIZATION_METRIC = "recall"      # Best model metric
MODEL_THRESHOLD = 0.5                     # Classification threshold
MODEL_PATH = None                         # Optional: direct model file path
```

**Load from .env**:
```env
MLFLOW_EXPERIMENT_NAME=Colab_GPU_Training
MODEL_OPTIMIZATION_METRIC=recall
MODEL_THRESHOLD=0.5
```

**Why**: 12-Factor App principle - separate config from code
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Pydantic Models
st.markdown("""
### 📝 Pydantic Models (Data Validation)

FastAPI uses Pydantic for automatic request/response validation:
""")

st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">🔹 CustomerData Model</div>', unsafe_allow_html=True
)
st.markdown(
    """
**Purpose**: Validate incoming customer data

**Fields** (all required):
```python
customerid: str                    # Customer identifier
age: int                          # Customer age
gender: str                       # "Male" or "Female"
tenure: int                       # Months with company
usage_frequency: int              # Monthly usage count
support_calls: int                # Support tickets
payment_delay: int                # Days payment delayed
subscription_type: str            # Basic/Standard/Premium
contract_length: str              # Monthly/Quarterly/Annual
total_spend: float                # Total spending ($)
last_interaction: int             # Days since last contact
```

**Example**:
```json
{
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
  "last_interaction": 10
}
```

**Validation**: FastAPI auto-checks types and required fields!
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">🔹 BatchPredictionRequest Model</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
**Purpose**: Validate batch prediction requests

**Fields**:
```python
customers: list[CustomerData]     # List of customer data
```

**Example**:
```json
{
  "customers": [
    {"customerid": "CUST001", "age": 35, ...},
    {"customerid": "CUST002", "age": 28, ...},
    {"customerid": "CUST003", "age": 45, ...}
  ]
}
```
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">🔹 PredictionResponse Model</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
**Purpose**: Structure single prediction response

**Fields**:
```python
customerid: str                   # Customer identifier
churn_probability: float          # 0.0 to 1.0
churn_prediction: int             # 0 or 1
risk_level: str                   # Low/Medium/High/Critical
```

**Example**:
```json
{
  "customerid": "CUST001",
  "churn_probability": 0.2345,
  "churn_prediction": 0,
  "risk_level": "Low"
}
```
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# API Endpoints
st.markdown("""
## 🌐 API Endpoints

All endpoints with their purposes and usage:
""")

# Endpoint 1: Root
st.markdown('<div class="endpoint-box">', unsafe_allow_html=True)
st.markdown('<div class="endpoint-name">📍 GET /</div>', unsafe_allow_html=True)
st.markdown(
    """
**Method**: <span class="code-method">GET</span>

**Purpose**: Welcome message and API information

**Response**:
```json
{
  "message": "Churn Prediction API",
  "version": "1.0.0",
  "status": "running"
}
```

**Use Case**: Quick check if API is accessible
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Endpoint 2: Health
st.markdown('<div class="endpoint-box">', unsafe_allow_html=True)
st.markdown('<div class="endpoint-name">📍 GET /health</div>', unsafe_allow_html=True)
st.markdown(
    """
**Method**: <span class="code-method">GET</span>

**Purpose**: Health check for AWS ECS/load balancers

**Response** (healthy):
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

**Response** (unhealthy):
```json
{
  "status": "unhealthy",
  "model_loaded": false
}
```

**Use Case**: 
- AWS ECS health checks
- Load balancer target health
- Monitoring/alerting
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Endpoint 3: Ready
st.markdown('<div class="endpoint-box">', unsafe_allow_html=True)
st.markdown('<div class="endpoint-name">📍 GET /ready</div>', unsafe_allow_html=True)
st.markdown(
    """
**Method**: <span class="code-method">GET</span>

**Purpose**: Readiness check (more detailed than health)

**Response**:
```json
{
  "ready": true,
  "model_info": {
    "source": "mlflow_experiment",
    "experiment_name": "Colab_GPU_Training",
    "run_id": "096c396d6abf4a7da6bae24acb8d99fe",
    "feature_count": 45,
    "threshold": 0.5
  }
}
```

**Use Case**: 
- Kubernetes readiness probes
- Verify model is loaded correctly
- Debug deployment issues
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Endpoint 4: Predict
st.markdown('<div class="endpoint-box">', unsafe_allow_html=True)
st.markdown('<div class="endpoint-name">📍 POST /predict</div>', unsafe_allow_html=True)
st.markdown(
    """
**Method**: <span class="code-method">POST</span>

**Purpose**: Single customer churn prediction

**Request Body**:
```json
{
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
  "last_interaction": 10
}
```

**Response**:
```json
{
  "customerid": "CUST001",
  "churn_probability": 0.2345,
  "churn_prediction": 0,
  "risk_level": "Low"
}
```

**cURL Example**:
```bash
curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "customerid": "CUST001",
    "age": 35,
    "gender": "Male",
    ...
  }'
```

**Python Example**:
```python
import requests

customer = {
    "customerid": "CUST001",
    "age": 35,
    ...
}

response = requests.post("http://localhost:8000/predict", json=customer)
result = response.json()
print(f"Risk: {result['risk_level']}")
```
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Endpoint 5: Batch Predict
st.markdown('<div class="endpoint-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="endpoint-name">📍 POST /predict/batch</div>', unsafe_allow_html=True
)
st.markdown(
    """
**Method**: <span class="code-method">POST</span>

**Purpose**: Batch prediction for multiple customers

**Request Body**:
```json
{
  "customers": [
    {
      "customerid": "CUST001",
      "age": 35,
      ...
    },
    {
      "customerid": "CUST002",
      "age": 28,
      ...
    }
  ]
}
```

**Response**:
```json
{
  "predictions": [
    {
      "customerid": "CUST001",
      "churn_probability": 0.2345,
      "churn_prediction": 0,
      "risk_level": "Low"
    },
    {
      "customerid": "CUST002",
      "churn_probability": 0.8521,
      "churn_prediction": 1,
      "risk_level": "Critical"
    }
  ],
  "total_customers": 2,
  "high_risk_count": 1,
  "churn_rate": 0.5
}
```

**Why Use Batch**:
- Much faster for multiple predictions (vectorized)
- Efficient resource usage
- Get aggregate statistics
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Endpoint 6: Model Info
st.markdown('<div class="endpoint-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="endpoint-name">📍 GET /model/info</div>', unsafe_allow_html=True
)
st.markdown(
    """
**Method**: <span class="code-method">GET</span>

**Purpose**: Get model metadata

**Response**:
```json
{
  "source": "mlflow_experiment",
  "experiment_name": "Colab_GPU_Training",
  "run_id": "096c396d6abf4a7da6bae24acb8d99fe",
  "feature_count": 45,
  "threshold": 0.5,
  "model_type": "XGBClassifier",
  "model_loaded": true
}
```

**Use Case**: 
- Verify correct model is loaded
- Debug production issues
- Model governance/tracking
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Lifespan Management
st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">🔄 Lifespan Management</div>', unsafe_allow_html=True
)
st.markdown(
    """
**Purpose**: Load model once at startup, cleanup at shutdown

**Code**:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model once
    global predictor
    predictor = create_predictor_from_mlflow(
        experiment_name=config.MLFLOW_EXPERIMENT_NAME,
        metric=config.MODEL_OPTIMIZATION_METRIC,
        threshold=config.MODEL_THRESHOLD
    )
    logger.info("Model loaded successfully")
    
    yield  # API runs here
    
    # Shutdown: Cleanup
    logger.info("Shutting down...")
```

**Why Important**:
- Model loads ONCE at startup (not per request)
- Fast predictions (model already in memory)
- Proper cleanup on shutdown
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# FILE: Dockerfile
# ============================================================================

st.markdown(
    '<div class="file-header">🐳 File: Dockerfile</div>', unsafe_allow_html=True
)

st.markdown("""
**Purpose**: Build production Docker image

**Location**: `Dockerfile` (project root)

**Strategy**: Multi-stage build for optimization
""")

st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">📦 Stage 1: Builder</div>', unsafe_allow_html=True
)
st.markdown(
    """
**Purpose**: Install dependencies

```dockerfile
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc g++ \\
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
```

**Why Separate Stage**:
- Build tools not needed in production
- Keeps final image small
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">🚀 Stage 2: Production</div>', unsafe_allow_html=True
)
st.markdown(
    """
**Purpose**: Final lightweight image

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy only installed packages (not build tools)
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY main.py .
COPY src/ src/

# Copy MLflow artifacts
COPY mlruns/ mlruns/

# Non-root user for security
RUN useradd -m -u 1000 appuser
USER appuser

# Environment
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s \\
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Features**:
- 🔒 Non-root user (security)
- 🏥 HEALTHCHECK for Docker/ECS
- 📦 Only production files copied
- 🚀 Fast startup with uvicorn
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# FILE: docker-compose.yml
# ============================================================================

st.markdown(
    '<div class="file-header">🐙 File: docker-compose.yml</div>', unsafe_allow_html=True
)

st.markdown("""
**Purpose**: Local testing environment

**Location**: `docker-compose.yml` (project root)
""")

st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">🧪 Local Testing Setup</div>', unsafe_allow_html=True
)
st.markdown(
    """
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=./mlruns
      - MLFLOW_EXPERIMENT_NAME=Colab_GPU_Training
      - MODEL_OPTIMIZATION_METRIC=recall
      - MODEL_THRESHOLD=0.5
    volumes:
      - ./mlruns:/app/mlruns  # Mount MLflow artifacts
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 40s
```

**Usage**:
```bash
# Build and start
docker-compose up --build

# Access API
curl http://localhost:8000

# Stop
docker-compose down
```
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# FILE: .dockerignore
# ============================================================================

st.markdown(
    '<div class="file-header">🚫 File: .dockerignore</div>', unsafe_allow_html=True
)

st.markdown("""
**Purpose**: Exclude files from Docker build context

**Location**: `.dockerignore` (project root)

**Why Important**: Smaller build context = faster builds
""")

st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">📋 Excluded Files</div>', unsafe_allow_html=True
)
st.markdown(
    """
```
# Version control
.git
.gitignore

# Python
__pycache__
*.pyc
*.pyo
.pytest_cache
.venv

# Development
notebooks/
tests/
*.ipynb
.env

# Documentation
README.md
*.md

# IDE
.vscode
.idea
```

**Result**: Faster builds, smaller images
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# FILE: ecs-task-definition.json
# ============================================================================

st.markdown(
    '<div class="file-header">☁️ File: ecs-task-definition.json</div>',
    unsafe_allow_html=True,
)

st.markdown("""
**Purpose**: AWS ECS Fargate deployment configuration

**Location**: `ecs-task-definition.json` (project root)
""")

st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">⚙️ ECS Configuration</div>', unsafe_allow_html=True
)
st.markdown(
    """
**Key Settings**:

```json
{
  "family": "churn-prediction-api",
  "networkMode": "awsvpc",  # Fargate requires this
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",             # 0.5 vCPU
  "memory": "1024",         # 1 GB RAM
  
  "containerDefinitions": [{
    "name": "churn-api",
    "image": "<account>.dkr.ecr.<region>.amazonaws.com/churn-api:latest",
    "portMappings": [{
      "containerPort": 8000,
      "protocol": "tcp"
    }],
    "environment": [
      {"name": "MLFLOW_EXPERIMENT_NAME", "value": "Colab_GPU_Training"},
      {"name": "MODEL_OPTIMIZATION_METRIC", "value": "recall"}
    ],
    "healthCheck": {
      "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
      "interval": 30,
      "timeout": 5,
      "retries": 3,
      "startPeriod": 60
    }
  }]
}
```

**Deployment Steps**:
1. Build Docker image
2. Push to Amazon ECR
3. Register task definition
4. Create/update ECS service
5. Configure load balancer
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Deployment Flow
st.markdown("""
## 🚀 Complete Deployment Flow

### **Local Development**
```bash
# 1. Test locally
python main.py

# 2. Test with Docker
docker-compose up --build

# 3. Test endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -d '{"customerid": "CUST001", ...}'
```

### **AWS ECS Deployment**
```bash
# 1. Build image
docker build -t churn-api:latest .

# 2. Tag for ECR
docker tag churn-api:latest <account>.dkr.ecr.<region>.amazonaws.com/churn-api:latest

# 3. Login to ECR
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com

# 4. Push to ECR
docker push <account>.dkr.ecr.<region>.amazonaws.com/churn-api:latest

# 5. Register task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json

# 6. Update service
aws ecs update-service --cluster churn-cluster --service churn-api-service --force-new-deployment
```

### **Monitoring**
- **CloudWatch Logs**: Application logs
- **ECS Metrics**: CPU, memory usage
- **Health Checks**: /health endpoint
- **ALB Metrics**: Request count, latency

## 🎯 Key Takeaways

### **FastAPI (main.py)**
- 🚀 Fast async web framework
- ✅ Auto validation with Pydantic
- 📚 Auto-generated docs at /docs
- 🔄 Lifespan management for model loading

### **Docker**
- 🐳 Multi-stage builds for optimization
- 🔒 Non-root user for security
- 🏥 Health checks for orchestration
- 📦 Consistent environments (dev = prod)

### **AWS ECS**
- ☁️ Fargate = serverless containers
- 🔄 Auto-scaling based on load
- 🏥 Health checks + auto-recovery
- 🌐 Load balancer integration

### **Production Ready Features**
- ✅ Health/readiness endpoints
- ✅ Error handling and logging
- ✅ CORS configuration
- ✅ Environment-based config
- ✅ Batch processing support

**Next Section**: 📊 Project Overview (complete project structure!)
""")

st.markdown("---")
st.markdown(
    "👉 **Navigate to 'Project Overview' in the sidebar to complete the documentation**"
)
