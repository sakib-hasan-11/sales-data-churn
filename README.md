(Project README)

# 🔮 Customer Churn Prediction - End-to-End ML Project

A production-ready machine learning system for predicting customer churn with complete MLOps pipeline, including data validation, feature engineering, model training, hyperparameter tuning, REST API, interactive UI, and containerized deployment.

## 📋 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Architecture](#-project-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
  - [1. Data Processing](#1-data-processing)
  - [2. Model Training](#2-model-training)
  - [3. Model Inference](#3-model-inference)
  - [4. API Service](#4-api-service)
  - [5. Streamlit UI](#5-streamlit-ui)
  - [6. Docker Deployment](#6-docker-deployment)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [MLflow Tracking](#-mlflow-tracking)
- [AWS S3 Integration](#-aws-s3-integration)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

---

## ✨ Features

### Core ML Features
- **Automated Data Processing Pipeline** - Load, validate, and preprocess data
- **Data Validation** - Great Expectations integration for data quality checks
- **Feature Engineering** - Automated feature extraction and transformation
- **Hyperparameter Tuning** - Optuna-based optimization
- **Model Training** - XGBoost with imbalanced-learn techniques
- **Experiment Tracking** - MLflow for model versioning and metrics
- **Model Evaluation** - Comprehensive metrics (Precision, Recall, F1, ROC-AUC)

### Production Features
- **REST API** - FastAPI-based production API with health checks
- **Interactive UI** - Streamlit dashboard for predictions and monitoring
- **Containerization** - Docker and Docker Compose for easy deployment
- **Cloud Integration** - AWS S3 for model storage and retrieval
- **Load Testing** - Locust integration for performance testing
- **Comprehensive Testing** - Unit tests, integration tests, and edge case handling

### MLOps Features
- **CI/CD Ready** - Automated testing and deployment workflows
- **Model Versioning** - MLflow model registry
- **Model Deployment Strategies** - File-based, MLflow, or S3 model loading
- **Monitoring & Health Checks** - API health endpoints and logging
- **Scalable Architecture** - Microservices design with Docker

---

## 🛠 Tech Stack

**Machine Learning & Data Science:**
- Python 3.11+
- scikit-learn, XGBoost
- Pandas, NumPy, SciPy
- Optuna (hyperparameter tuning)
- imbalanced-learn
- category-encoders

**MLOps & Experiment Tracking:**
- MLflow (experiment tracking, model registry)
- Great Expectations (data validation)
- Pydantic (data validation)

**API & Web:**
- FastAPI (REST API)
- Uvicorn (ASGI server)
- Streamlit (interactive UI)
- Plotly, Matplotlib, Seaborn (visualization)

**DevOps & Cloud:**
- Docker & Docker Compose
- AWS S3 (model storage)
- boto3 (AWS SDK)
- GitHub Actions (CI/CD)

**Testing & Quality:**
- pytest, pytest-cov
- Locust (load testing)
- pytest-benchmark

---

## 🏗 Project Architecture

```
┌─────────────────┐
│   Raw Data      │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Data Validation │ ← Great Expectations
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Preprocessing  │
└────────┬────────┘
         │
         v
┌─────────────────┐
│Feature Engineer │
└────────┬────────┘
         │
         v
┌─────────────────┐
│Hyperparameter   │ ← Optuna
│    Tuning       │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Model Training  │ ← MLflow Tracking
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Model Storage  │ → File / MLflow / S3
└────────┬────────┘
         │
         v
    ┌────┴────┐
    │         │
    v         v
┌──────┐  ┌──────────┐
│ API  │  │Streamlit │
└──────┘  └──────────┘
```

---

## 📦 Prerequisites

- **Python 3.11 or higher**
- **pip** or **conda** package manager
- **Docker** and **Docker Compose** (for containerized deployment)
- **Git** (for cloning the repository)
- **AWS Account** (optional, for S3 integration)

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd sales-data-churn
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install Streamlit dependencies (if using UI)
pip install -r requirements_streamlit.txt
```

### 4. Verify Installation

```bash
python -c "import sklearn, xgboost, mlflow, fastapi; print('All dependencies installed successfully!')"
```

---

## ⚡ Quick Start

### Option 1: Run Complete Pipeline (Recommended for First Time)

```bash
# Run the full production pipeline
python scripts/run_pipeline.py
```

This will:
1. Load and validate data
2. Preprocess and engineer features
3. Tune hyperparameters with Optuna
4. Train multiple models with MLflow
5. Evaluate and save the best model

### Option 2: Quick Testing Pipeline

```bash
# Fast execution for testing
python scripts/quick_pipeline.py
```

### Option 3: Modular Execution

```bash
# Run specific stages only
python scripts/modular_pipeline.py --stages load preprocess features train

# Run from a specific stage onwards
python scripts/modular_pipeline.py --from features
```

---

## 📖 Usage Guide

### 1. Data Processing

#### Prepare Your Data

Place your data files in the `data/raw/` directory:
- `train.csv` - Training data
- `test.csv` - Test data
- `holdout.csv` - Holdout/validation data

#### Validate Data

```bash
# Validate data quality with Great Expectations
python scripts/validate_train_ge.py --csv data/raw/train.csv \
    --suite great_expectations/expectations/train_suite.yml
```

#### Preprocess Data

```python
from src.data_processing.load import load_data
from src.data_processing.preprocess import preprocess_data

# Load data
train_df = load_data("data/raw/train.csv")

# Preprocess
train_processed = preprocess_data(train_df)
```

### 2. Model Training

#### Using the Pipeline Scripts

```bash
# Full pipeline with all features
python scripts/run_pipeline.py

# Quick pipeline for testing
python scripts/quick_pipeline.py

# Custom configuration
python scripts/modular_pipeline.py --all --n-trials 100 --n-runs 5 --threshold 0.5
```

#### Manual Training

```python
from src.training.optuna_tuning import optimize_hyperparameters
from src.training.mlflow_training import train_with_mlflow

# Optimize hyperparameters
best_params = optimize_hyperparameters(X_train, y_train, n_trials=50)

# Train with MLflow
run_id = train_with_mlflow(X_train, y_train, X_test, y_test, best_params)
```

### 3. Model Inference

#### Single Prediction

```python
from src.inference.inference import create_predictor_from_file

# Load predictor
predictor = create_predictor_from_file("models/churn_model_production.pkl")

# Make prediction
customer_data = {
    "AccountWeeks": 100,
    "ContractRenewal": 1,
    "DataPlan": 1,
    "DataUsage": 2.5,
    # ... other features
}

result = predictor.predict_single(customer_data)
print(f"Churn Probability: {result['churn_probability']:.2%}")
print(f"Prediction: {result['prediction']}")
```

#### Batch Prediction

```python
# Predict on DataFrame
predictions = predictor.predict_batch(test_df)
```

### 4. API Service

#### Start the API Server

```bash
# Development mode
python main.py

# Production mode with Uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Configure Model Source

```bash
# Use file-based model
export MODEL_SOURCE=file
export MODEL_PATH=models/churn_model_production.pkl

# Use MLflow model
export MODEL_SOURCE=mlflow
export MLFLOW_TRACKING_URI=./mlruns
export MLFLOW_EXPERIMENT_NAME=Colab_GPU_Training

# Use S3 model
export MODEL_SOURCE=s3
export S3_BUCKET_NAME=your-bucket-name
export S3_MODEL_NAME=churn_model_production.pkl
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
```

#### Test API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"AccountWeeks": 100, "ContractRenewal": 1, "DataPlan": 1, "DataUsage": 2.5, ...}'

# Batch prediction
python test_api.py
```

### 5. Streamlit UI

#### Start Streamlit Dashboard

```bash
# Make sure API is running first
python main.py

# In another terminal, start Streamlit
streamlit run streamlit_app.py
```

#### Access the UI

Open your browser to `http://localhost:8501`

**Features:**
- Single customer prediction
- Batch predictions via CSV upload
- Model information and metrics
- Feature importance visualization
- Interactive explanations

#### Using Documentation Pages

```bash
# Start the comprehensive documentation app
streamlit run streamlit_docs/Home.py
```

This includes:
- 📁 Data Processing Guide
- 🔧 Feature Engineering
- 🎓 Model Training
- 🔮 Inference Engine
- 🌐 API Deployment
- 📊 Project Overview
- 🚀 CI/CD Deployment

### 6. Docker Deployment

#### Build and Run with Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up -d

# Stop services
docker-compose down
```

This will start:
- **API Service**: `http://localhost:8000`
- **Streamlit UI**: `http://localhost:8501`

#### Build Individual Containers

```bash
# Build API container
docker build -t churn-api -f Dockerfile .

# Build Streamlit container
docker build -t churn-frontend -f Dockerfile.streamlit .

# Run API container
docker run -p 8000:8000 churn-api

# Run Streamlit container
docker run -p 8501:8501 churn-frontend
```

#### Environment Variables for Docker

Create a `.env` file:

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
S3_BUCKET_NAME=churn-project-model
S3_MODEL_NAME=churn_model_production.pkl

# Model Configuration
MODEL_SOURCE=s3
MLFLOW_TRACKING_URI=./mlruns
MLFLOW_EXPERIMENT_NAME=Colab_GPU_Training

# API Configuration
DEBUG=false
API_HOST=0.0.0.0
API_PORT=8000
```

---

## 📂 Project Structure

```
sales-data-churn/
├── data/                          # Data directory
│   ├── raw/                       # Raw data files
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── holdout.csv
│   └── processed/                 # Processed data files
│       ├── train_processed.csv
│       ├── test_processed.csv
│       └── holdout_processed.csv
│
├── src/                           # Source code
│   ├── data_processing/           # Data loading and preprocessing
│   │   ├── load.py
│   │   └── preprocess.py
│   ├── features/                  # Feature engineering
│   │   ├── build_feature.py
│   │   └── feature_preprocess.py
│   ├── training/                  # Model training
│   │   ├── optuna_tuning.py
│   │   ├── mlflow_training.py
│   │   └── evaluation.py
│   ├── inference/                 # Model inference
│   │   └── inference.py
│   └── utils/                     # Utility functions
│       ├── data_validator.py
│       ├── s3_handler.py
│       └── upload_to_s3.py
│
├── scripts/                       # Pipeline scripts
│   ├── run_pipeline.py            # Full production pipeline
│   ├── quick_pipeline.py          # Fast testing pipeline
│   ├── modular_pipeline.py        # Flexible stage execution
│   ├── colab_pipeline.py          # Google Colab pipeline
│   ├── colab_evaluate_holdout.py  # Holdout evaluation
│   └── README.md                  # Scripts documentation
│
├── tests/                         # Test suite
│   ├── test_all.py                # Comprehensive tests
│   ├── test_data_processing.py
│   ├── test_feature_engineering.py
│   ├── test_inference.py
│   ├── test_edge_cases.py
│   └── test_performance.py
│
├── streamlit_docs/                # Streamlit documentation
│   ├── Home.py                    # Documentation home
│   └── pages/                     # Documentation pages
│       ├── 1_📁_Data_Processing.py
│       ├── 2_🔧_Feature_Engineering.py
│       ├── 3_🎓_Model_Training.py
│       ├── 4_🔮_Inference_Engine.py
│       ├── 5_🌐_API_Deployment.py
│       ├── 6_📊_Project_Overview.py
│       └── 7_🚀_CI_CD_Deployment.py
│
├── notebooks/                     # Jupyter notebooks
│   ├── EDA_and_Feature_Engineering.ipynb
│   ├── tree_based_recall_models.ipynb
│   └── Colab_GPU_Training.ipynb
│
├── mlruns/                        # MLflow tracking data
├── models/                        # Saved models
├── outputs/                       # Evaluation outputs
│
├── main.py                        # FastAPI application
├── streamlit_app.py               # Streamlit UI
├── test_api.py                    # API testing script
│
├── Dockerfile                     # API Docker image
├── Dockerfile.streamlit           # Streamlit Docker image
├── docker-compose.yml             # Docker Compose configuration
│
├── requirements.txt               # Python dependencies
├── requirements_streamlit.txt     # Streamlit dependencies
├── pyproject.toml                 # Project metadata
└── README.md                      # This file
```

---

## ⚙️ Configuration

### Pipeline Configuration

Edit `scripts/run_pipeline.py` to customize the pipeline:

```python
class PipelineConfig:
    # Data paths
    TRAIN_PATH: str = "data/raw/train.csv"
    TEST_PATH: str = "data/raw/test.csv"
    HOLDOUT_PATH: str = "data/raw/holdout.csv"
    
    # Optuna settings
    N_TRIALS: int = 50
    
    # MLflow settings
    N_RUNS: int = 3
    EXPERIMENT_NAME: str = "Colab_GPU_Training"
    
    # Model settings
    OPTIMIZATION_METRIC: str = "recall"
    THRESHOLD: float = 0.5
    PREPROCESS_STRATEGY: str = "yeo-johnson"
```

### API Configuration

Set environment variables or edit `.env`:

```bash
# Application
APP_NAME=Churn Prediction API
APP_VERSION=1.0.0
DEBUG=false

# Server
API_HOST=0.0.0.0
API_PORT=8000

# Model source (file, mlflow, or s3)
MODEL_SOURCE=mlflow

# File-based model
MODEL_PATH=models/churn_model_production.pkl

# MLflow settings
MLFLOW_TRACKING_URI=./mlruns
MLFLOW_EXPERIMENT_NAME=Colab_GPU_Training
MLFLOW_MODEL_NAME=best_model
MLFLOW_RUN_ID=<run-id>

# S3 settings
S3_BUCKET_NAME=churn-project-model
S3_MODEL_NAME=churn_model_production.pkl
S3_REGION=us-east-1
AWS_ACCESS_KEY_ID=<your-key>
AWS_SECRET_ACCESS_KEY=<your-secret>
```

---

## 🧪 Testing

### Run All Tests

```bash
# Run complete test suite
pytest tests/

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_inference.py -v

# Run with detailed output
pytest tests/ -v -s
```

### Test Categories

```bash
# Data processing tests
pytest tests/test_data_processing.py

# Feature engineering tests
pytest tests/test_feature_engineering.py

# Inference tests
pytest tests/test_inference.py

# Edge cases
pytest tests/test_edge_cases.py

# Performance tests
pytest tests/test_performance.py
```

### Load Testing

```bash
# Install Locust
pip install locust

# Run load tests (make sure API is running)
locust -f tests/locustfile.py --host=http://localhost:8000
```

Open `http://localhost:8089` to access Locust UI.

---

## 📊 MLflow Tracking

### Start MLflow UI

```bash
# Start MLflow server
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Access at `http://localhost:5000`

### Track Experiments

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("./mlruns")

# Set experiment
mlflow.set_experiment("Colab_GPU_Training")

# Start run
with mlflow.start_run(run_name="xgboost_run"):
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    
    # Log metrics
    mlflow.log_metric("recall", 0.85)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

### View Experiments

The MLflow UI shows:
- All experiment runs
- Parameters and metrics
- Model artifacts
- Comparison between runs
- Model registry

---

## ☁️ AWS S3 Integration

### Upload Model to S3

```python
from src.utils.upload_to_s3 import upload_model_to_s3

# Upload model
upload_model_to_s3(
    local_file_path="models/churn_model_production.pkl",
    bucket_name="churn-project-model",
    s3_file_name="churn_model_production.pkl"
)
```

### Download Model from S3

```python
from src.utils.s3_handler import S3Handler

# Initialize handler
s3_handler = S3Handler(bucket_name="churn-project-model")

# Download model
s3_handler.download_model("churn_model_production.pkl", "models/downloaded_model.pkl")
```

### Configure AWS Credentials

```bash
# Option 1: Environment variables
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_REGION=us-east-1

# Option 2: AWS CLI
aws configure

# Option 3: .env file
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

The project includes CI/CD workflows for:

1. **Automated Testing**
   - Run on every push/PR
   - Execute full test suite
   - Generate coverage reports

2. **Docker Build**
   - Build Docker images
   - Push to container registry
   - Deploy to staging/production

3. **Model Deployment**
   - Validate model performance
   - Upload to S3
   - Update API service

### Setup CI/CD

1. Add secrets to GitHub repository:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `S3_BUCKET_NAME`

2. Workflow files are in `.github/workflows/`

3. Customize deployment targets in workflow files

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Make sure src is in Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

# Or add to script
import sys
sys.path.insert(0, "src")
```

#### 2. MLflow Tracking Issues

```bash
# Set tracking URI explicitly
export MLFLOW_TRACKING_URI=./mlruns

# Or in Python
import mlflow
mlflow.set_tracking_uri("./mlruns")
```

#### 3. Missing Dependencies

```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall

# Install specific package
pip install <package-name>
```

#### 4. Docker Permission Issues

```bash
# On Linux, add user to docker group
sudo usermod -aG docker $USER

# Restart Docker service
sudo systemctl restart docker
```

#### 5. API Connection Issues

```bash
# Check if API is running
curl http://localhost:8000/health

# Check Docker logs
docker logs churn-api

# Restart services
docker-compose restart
```

#### 6. Model Loading Errors

```bash
# Verify model file exists
ls -lh models/

# Check model format
python -c "import joblib; model = joblib.load('models/churn_model_production.pkl'); print(type(model))"

# Re-download from S3 if needed
python -c "from src.utils.s3_handler import S3Handler; S3Handler('bucket-name').download_model('model.pkl', 'models/model.pkl')"
```

### Getting Help

- Check the [scripts/README.md](scripts/README.md) for pipeline documentation
- Review logs in `logs/` directory
- Check MLflow UI for experiment details
- Inspect Docker logs: `docker-compose logs`

---

## 🤝 Contributing

Contributions are welcome! Here's how to contribute:

1. **Fork the repository**

2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make your changes**
   - Write clean, documented code
   - Follow existing code style
   - Add tests for new features

4. **Run tests**
   ```bash
   pytest tests/ --cov=src
   ```

5. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```

6. **Push to your branch**
   ```bash
   git push origin feature/amazing-feature
   ```

7. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guide
- Write docstrings for all functions
- Add type hints where appropriate
- Update tests for changes
- Update documentation

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📧 Contact

For questions or support, please open an issue on the GitHub repository.

---

## 🎯 Roadmap

Future enhancements planned:

- [ ] Add more ML models (Random Forest, LightGBM, CatBoost)
- [ ] Implement model explainability (SHAP, LIME)
- [ ] Add real-time monitoring dashboard
- [ ] Kubernetes deployment manifests
- [ ] A/B testing framework
- [ ] Model drift detection
- [ ] Automated retraining pipeline
- [ ] Enhanced feature store integration
- [ ] GraphQL API support
- [ ] Mobile app integration

---

## ⭐ Acknowledgments

- **MLflow** for experiment tracking
- **Optuna** for hyperparameter optimization
- **FastAPI** for high-performance API
- **Streamlit** for rapid UI development
- **Great Expectations** for data validation
- **XGBoost** for powerful gradient boosting

---

**Made with ❤️ for production-ready ML systems**

