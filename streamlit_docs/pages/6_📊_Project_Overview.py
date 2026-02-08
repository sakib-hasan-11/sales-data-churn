"""
Page 6: Project Overview
Complete project structure and flow documentation
"""

import streamlit as st

st.set_page_config(page_title="Project Overview", page_icon="📊", layout="wide")

# Custom CSS
st.markdown(
    """
<style>
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #6A1B9A;
        margin-top: 2rem;
        background-color: #F3E5F5;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #6A1B9A;
    }
    .module-box {
        background-color: #E8EAF6;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #3F51B5;
    }
    .flow-box {
        background-color: #E0F2F1;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #00897B;
    }
    .file-tree {
        font-family: 'Courier New', monospace;
        background-color: #263238;
        color: #AED581;
        padding: 1.5rem;
        border-radius: 8px;
        font-size: 0.9rem;
        line-height: 1.6;
        overflow-x: auto;
    }
    .highlight {
        background-color: #FFECB3;
        padding: 2px 6px;
        border-radius: 3px;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.title("📊 Project Overview")
st.markdown("### Complete End-to-End ML Project Structure")
st.markdown("---")

# Project Summary
st.markdown("""
## 🎯 Project Summary

**Goal**: Predict customer churn with production-ready ML pipeline

**Business Value**:
- 📉 Reduce customer churn by identifying at-risk customers
- 💰 Enable proactive retention campaigns
- 📊 Optimize marketing spend with targeted interventions

**Tech Stack**:
- **ML**: XGBoost, Scikit-learn, Optuna
- **Tracking**: MLflow
- **API**: FastAPI, Uvicorn
- **Deployment**: Docker, AWS ECS
- **Data**: Pandas, NumPy
- **Validation**: Great Expectations
""")

st.markdown("---")

# Complete File Structure
st.markdown(
    '<div class="section-header">📁 Complete File Structure</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="file-tree">
sales-data-churn/
│
├── <span style="color: #80CBC4;">📄 main.py</span>                    # FastAPI application (production API)
├── <span style="color: #80CBC4;">📄 app.py</span>                     # Streamlit dashboard (existing)
├── <span style="color: #80CBC4;">📄 requirements.txt</span>           # Python dependencies
├── <span style="color: #80CBC4;">📄 Dockerfile</span>                 # Container image definition
├── <span style="color: #80CBC4;">📄 docker-compose.yml</span>         # Local Docker environment
├── <span style="color: #80CBC4;">📄 .dockerignore</span>              # Docker build exclusions
├── <span style="color: #80CBC4;">📄 .env.example</span>               # Configuration template
├── <span style="color: #80CBC4;">📄 ecs-task-definition.json</span>  # AWS ECS deployment config
├── <span style="color: #80CBC4;">📄 DEPLOYMENT.md</span>              # Deployment guide
├── <span style="color: #80CBC4;">📄 API_EXAMPLES.md</span>            # API usage examples
├── <span style="color: #80CBC4;">📄 test_api.py</span>                # API test suite
│
├── <span style="color: #FFD54F;">📂 data/</span>                      # Data storage
│   ├── raw/                  # Original datasets
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── holdout.csv
│   └── processed/            # Preprocessed datasets
│       ├── train_processed.csv
│       ├── test_processed.csv
│       └── holdout_processed.csv
│
├── <span style="color: #FFD54F;">📂 src/</span>                       # Source code (modular)
│   ├── __init__.py
│   │
│   ├── <span style="color: #81C784;">📂 data_processing/</span>    # Data loading & cleaning
│   │   ├── __init__.py
│   │   ├── <span style="color: #AED581;">load.py</span>           # Load CSV data
│   │   └── <span style="color: #AED581;">preprocess.py</span>     # Clean & handle missing values
│   │
│   ├── <span style="color: #81C784;">📂 features/</span>           # Feature engineering
│   │   ├── __init__.py
│   │   ├── <span style="color: #AED581;">build_feature.py</span>  # Create 10 new features
│   │   └── <span style="color: #AED581;">feature_preprocess.py</span> # Encode & scale
│   │
│   ├── <span style="color: #81C784;">📂 training/</span>           # Model training
│   │   ├── __init__.py
│   │   ├── <span style="color: #AED581;">optuna_tuning.py</span>  # Hyperparameter optimization
│   │   ├── <span style="color: #AED581;">mlflow_training.py</span> # MLflow experiment tracking
│   │   └── <span style="color: #AED581;">evaluation.py</span>      # Model evaluation & saving
│   │
│   ├── <span style="color: #81C784;">📂 inference/</span>          # Production inference
│   │   ├── __init__.py
│   │   └── <span style="color: #AED581;">inference.py</span>       # Prediction engine
│   │
│   └── <span style="color: #81C784;">📂 utils/</span>              # Utilities
│       ├── __init__.py
│       └── <span style="color: #AED581;">data_validator.py</span>  # Data quality checks
│
├── <span style="color: #FFD54F;">📂 scripts/</span>                   # Pipeline scripts
│   ├── run_pipeline.py       # Main training pipeline
│   ├── colab_pipeline.py     # Colab-specific pipeline
│   └── prepare_ci_data.py    # Data preparation
│
├── <span style="color: #FFD54F;">📂 notebooks/</span>                 # Jupyter notebooks
│   ├── EDA_and_Feature_Engineering.ipynb
│   └── tree_based_recall_models.ipynb
│
├── <span style="color: #FFD54F;">📂 models/</span>                    # Saved models (optional)
│
├── <span style="color: #FFD54F;">📂 mlruns/</span>                    # MLflow tracking data
│   └── <experiment_id>/
│       └── <run_id>/
│           ├── artifacts/    # Model files
│           ├── metrics/      # Performance metrics
│           ├── params/       # Hyperparameters
│           └── tags/         # Metadata
│
├── <span style="color: #FFD54F;">📂 outputs/</span>                   # Evaluation results
│   ├── holdout_predictions.csv
│   └── holdout_evaluation_summary.txt
│
├── <span style="color: #FFD54F;">📂 tests/</span>                     # Unit tests
│   ├── __init__.py
│   └── test_data.py
│
└── <span style="color: #FFD54F;">📂 streamlit_docs/</span>            # This documentation!
    ├── Home.py
    └── pages/
        ├── 1_📁_Data_Processing.py
        ├── 2_🔧_Feature_Engineering.py
        ├── 3_🎓_Model_Training.py
        ├── 4_🔮_Inference_Engine.py
        ├── 5_🌐_API_Deployment.py
        └── 6_📊_Project_Overview.py
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("---")

# Module Summary
st.markdown(
    '<div class="section-header">📦 Module Summary</div>', unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="module-box">', unsafe_allow_html=True)
    st.markdown(
        """
    ### 📁 Data Processing
    **Files**: 3  
    **Functions**: 3
    
    - `load.py`: Load CSV data
    - `preprocess.py`: Clean & impute
    - `data_validator.py`: Quality checks
    
    **Purpose**: Prepare raw data
    """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="module-box">', unsafe_allow_html=True)
    st.markdown(
        """
    ### 🎓 Model Training
    **Files**: 3  
    **Functions**: 4
    
    - `optuna_tuning.py`: HPO with Optuna
    - `mlflow_training.py`: Track experiments
    - `evaluation.py`: Save best model
    
    **Purpose**: Train & optimize
    """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="module-box">', unsafe_allow_html=True)
    st.markdown(
        """
    ### 🌐 API Deployment
    **Files**: 5  
    **Endpoints**: 6
    
    - `main.py`: FastAPI app
    - `Dockerfile`: Container
    - `docker-compose.yml`: Local test
    - `ecs-task-definition.json`: AWS ECS
    - `.dockerignore`: Optimization
    
    **Purpose**: Production API
    """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="module-box">', unsafe_allow_html=True)
    st.markdown(
        """
    ### 🔧 Feature Engineering
    **Files**: 2  
    **Functions**: 2
    
    - `build_feature.py`: 10 new features
    - `feature_preprocess.py`: Encode & scale
    
    **Purpose**: Create ML features
    """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="module-box">', unsafe_allow_html=True)
    st.markdown(
        """
    ### 🔮 Inference Engine
    **Files**: 1  
    **Classes**: 3  
    **Functions**: 12+
    
    - `inference.py`: Production predictions
      - InferencePreprocessor
      - ModelLoader
      - ChurnPredictor
    
    **Purpose**: Make predictions
    """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Data Flow
st.markdown(
    '<div class="section-header">🔄 Complete Data Flow</div>', unsafe_allow_html=True
)

st.markdown('<div class="flow-box">', unsafe_allow_html=True)
st.markdown(
    """
### **Training Pipeline** 🎓

```
1️⃣ Raw Data (train.csv)
    │
    ├─> load_data()                    [load.py]
    │
2️⃣ Loaded DataFrame
    │
    ├─> raw_preprocess()               [preprocess.py]
    │   • Handle missing values
    │   • Clean outliers
    │
3️⃣ Clean DataFrame
    │
    ├─> build_feature()                [build_feature.py]
    │   • CLV, support_efficiency
    │   • payment_reliability, etc.
    │
4️⃣ Engineered DataFrame
    │
    ├─> preprocess_features()          [feature_preprocess.py]
    │   • Label encode gender
    │   • One-hot encode categoricals
    │
5️⃣ Model-Ready Features (X, y)
    │
    ├─> tune_hyperparameters()         [optuna_tuning.py]
    │   • 100 Optuna trials
    │   • Optimize recall
    │
6️⃣ Best Hyperparameters
    │
    ├─> train_with_mlflow()            [mlflow_training.py]
    │   • Train XGBoost
    │   • Log to MLflow
    │
7️⃣ Trained Model
    │
    └─> save_model_from_mlflow()       [evaluation.py]
        • Evaluate on test
        • Save best model

Result: Model in MLflow (mlruns/)
```
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="flow-box">', unsafe_allow_html=True)
st.markdown(
    """
### **Inference Pipeline** 🔮

```
1️⃣ New Customer Data (JSON/dict)
    │
    ├─> FastAPI Endpoint              [main.py]
    │   • Validate with Pydantic
    │   • /predict or /predict/batch
    │
2️⃣ Validated Data
    │
    ├─> InferencePreprocessor          [inference.py]
    │   • clean_column_names()
    │   • raw_preprocess()
    │   • build_feature()
    │   • encode_features()
    │   • align_features()
    │
3️⃣ Preprocessed Features (X)
    │
    ├─> ChurnPredictor                 [inference.py]
    │   • model.predict_proba(X)
    │   • Apply threshold
    │   • Calculate risk level
    │
4️⃣ Prediction Result
    │
    └─> FastAPI Response               [main.py]
        • Format as JSON
        • Return to client

Result: {
  "churn_probability": 0.75,
  "churn_prediction": 1,
  "risk_level": "High"
}
```
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Feature Engineering Flow
st.markdown(
    '<div class="section-header">🔧 Feature Engineering Details</div>',
    unsafe_allow_html=True,
)

st.markdown("""
### **10 Engineered Features**

| Feature | Formula | Business Logic |
|---------|---------|----------------|
| 1. **CLV** | `total_spend / tenure` | Customer Lifetime Value - spending rate |
| 2. **support_efficiency** | `support_calls / max(usage_frequency, 1)` | Support needed per usage |
| 3. **payment_reliability** | `1 / (1 + payment_delay)` | On-time payment score |
| 4. **engagement_score** | `usage_frequency / (1 + last_interaction)` | Recent activity level |
| 5. **value_to_company** | `total_spend / (1 + support_calls)` | Revenue vs support cost |
| 6. **normalized_tenure** | `tenure / max(tenure)` | Relative customer age |
| 7. **days_since_last_interaction** | `last_interaction` | Customer engagement metric |
| 8. **Tenure Category** | Bins: 0-12M, 12-24M, 24-36M, 36-48M, 48M+ | Customer lifecycle stage |
| 9. **Age Group** | Bins: 18-30, 30-40, 40-50, 50-60, 60+ | Demographic segment |
| 10. **Spend Category** | Bottom 33% (Low), Mid 34% (Medium), Top 33% (High) | Spending tier |

**Result**: Original 11 columns → 45+ features after encoding!
""")

st.markdown("---")

# Model Performance
st.markdown(
    '<div class="section-header">📈 Model Performance</div>', unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🎯 Recall", "~95%", help="Catches 95% of churners")
with col2:
    st.metric("✅ Precision", "~40-50%", help="Half of predictions are true")
with col3:
    st.metric("📊 F1 Score", "~56%", help="Harmonic mean")
with col4:
    st.metric("🔍 AUC", "~70%", help="Overall discrimination")

st.markdown("""
**Why Optimize Recall?**
- Business prefers catching all churners (even with false alarms)
- Cost of losing a customer > Cost of retention offer to loyal customer
- Better to be safe than sorry!

**Trade-off**: Higher recall → Lower precision (more false positives)
""")

st.markdown("---")

# Quick Reference
st.markdown(
    '<div class="section-header">🚀 Quick Reference Guide</div>', unsafe_allow_html=True
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["🎓 Training", "🔮 Inference", "🌐 API", "☁️ Deployment"]
)

with tab1:
    st.markdown("""
    ### Train New Model
    
    ```python
    # 1. Load and preprocess data
    from src.data_processing.load import load_data
    from src.data_processing.preprocess import raw_preprocess
    from src.features.build_feature import build_feature
    from src.features.feature_preprocess import preprocess_features
    
    df = load_data('data/raw/train.csv')
    df_clean = raw_preprocess(df)
    df_features = build_feature(df_clean)
    X, y, feature_names = preprocess_features(df_features)
    
    # 2. Tune hyperparameters
    from src.training.optuna_tuning import tune_hyperparameters
    
    best_params = tune_hyperparameters(X, y)
    
    # 3. Train with MLflow
    from src.training.mlflow_training import train_with_mlflow
    
    run_id = train_with_mlflow(
        X, y,
        params=best_params,
        experiment_name='My_Experiment'
    )
    
    print(f"Model trained! Run ID: {run_id}")
    ```
    """)

with tab2:
    st.markdown("""
    ### Make Predictions
    
    ```python
    from src.inference.inference import create_predictor_from_mlflow
    
    # Initialize predictor
    predictor = create_predictor_from_mlflow(
        experiment_name='Colab_GPU_Training',
        metric='recall'
    )
    
    # Single prediction
    customer = {
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
    
    result = predictor.predict(customer)
    print(result)
    # {'churn_probability': 0.23, 'churn_prediction': 0, 'risk_level': 'Low'}
    
    # Batch prediction
    customers = [customer1, customer2, ...]
    batch_result = predictor.predict_batch(customers)
    ```
    """)

with tab3:
    st.markdown("""
    ### Use API
    
    **Start Server**:
    ```bash
    uvicorn main:app --reload
    ```
    
    **Single Prediction**:
    ```python
    import requests
    
    customer = {
        "customerid": "CUST001",
        "age": 35,
        # ... other fields
    }
    
    response = requests.post(
        "http://localhost:8000/predict",
        json=customer
    )
    
    result = response.json()
    print(f"Risk: {result['risk_level']}")
    ```
    
    **Batch Prediction**:
    ```python
    payload = {"customers": [customer1, customer2, ...]}
    
    response = requests.post(
        "http://localhost:8000/predict/batch",
        json=payload
    )
    
    results = response.json()
    print(f"High risk: {results['high_risk_count']}")
    ```
    
    **Check Health**:
    ```bash
    curl http://localhost:8000/health
    ```
    """)

with tab4:
    st.markdown("""
    ### Deploy to AWS ECS
    
    **1. Build & Test Locally**:
    ```bash
    # Test with Docker Compose
    docker-compose up --build
    
    # Test endpoints
    curl http://localhost:8000/health
    ```
    
    **2. Push to ECR**:
    ```bash
    # Build
    docker build -t churn-api:latest .
    
    # Tag
    docker tag churn-api:latest <account>.dkr.ecr.<region>.amazonaws.com/churn-api:latest
    
    # Login
    aws ecr get-login-password --region <region> | \\
      docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
    
    # Push
    docker push <account>.dkr.ecr.<region>.amazonaws.com/churn-api:latest
    ```
    
    **3. Deploy to ECS**:
    ```bash
    # Register task definition
    aws ecs register-task-definition \\
      --cli-input-json file://ecs-task-definition.json
    
    # Update service
    aws ecs update-service \\
      --cluster churn-cluster \\
      --service churn-api-service \\
      --force-new-deployment
    ```
    
    **4. Configure Load Balancer**:
    - Target: ECS service
    - Health check: `/health`
    - Port: 8000
    """)

st.markdown("---")

# Project Statistics
st.markdown(
    '<div class="section-header">📊 Project Statistics</div>', unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📁 Code Structure
    - **Total Modules**: 8
    - **Total Functions**: 15+
    - **Total Classes**: 3
    - **API Endpoints**: 6
    - **Lines of Code**: ~2000+
    """)

with col2:
    st.markdown("""
    ### 🔧 Features
    - **Raw Features**: 11
    - **Engineered Features**: 10
    - **Total Model Features**: 45+
    - **Categorical Encoded**: 5
    - **Numerical Features**: 6
    """)

with col3:
    st.markdown("""
    ### 🚀 Deployment
    - **Docker Stages**: 2
    - **Container Size**: ~1GB
    - **API Response Time**: <100ms
    - **Batch Processing**: Yes
    - **AWS ECS Ready**: Yes
    """)

st.markdown("---")

# Technology Stack
st.markdown(
    '<div class="section-header">💻 Technology Stack</div>', unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### **Core ML**
    - 🤖 **XGBoost**: Gradient boosting classifier
    - 🔬 **Scikit-learn**: Preprocessing, metrics
    - 🎯 **Optuna**: Hyperparameter optimization
    - 📊 **MLflow**: Experiment tracking
    
    ### **Data Processing**
    - 🐼 **Pandas**: Data manipulation
    - 🔢 **NumPy**: Numerical operations
    - ✅ **Great Expectations**: Data validation
    
    ### **API & Deployment**
    - ⚡ **FastAPI**: Web framework
    - 🦄 **Uvicorn**: ASGI server
    - 🐳 **Docker**: Containerization
    - ☁️ **AWS ECS**: Orchestration
    """)

with col2:
    st.markdown("""
    ### **Key Libraries**
    ```
    fastapi==0.115.0
    uvicorn==0.30.6
    pydantic==2.9.2
    xgboost==2.1.1
    scikit-learn==1.5.2
    pandas==2.2.3
    numpy==1.26.4
    mlflow==2.16.2
    optuna==4.0.0
    great-expectations==1.1.3
    python-multipart==0.0.12
    joblib==1.4.2
    ```
    
    ### **Development Tools**
    - 🎨 **Streamlit**: Documentation
    - 📓 **Jupyter**: Exploration
    - 🧪 **Pytest**: Testing
    - 📝 **Markdown**: Documentation
    """)

st.markdown("---")

# Best Practices
st.markdown(
    '<div class="section-header">✨ Best Practices Implemented</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### **Code Quality**
    - ✅ Modular architecture (src/ structure)
    - ✅ Separation of concerns
    - ✅ Reusable components
    - ✅ Type hints with Pydantic
    - ✅ Error handling
    - ✅ Logging throughout
    
    ### **ML Best Practices**
    - ✅ Feature engineering pipeline
    - ✅ Hyperparameter tuning
    - ✅ Experiment tracking (MLflow)
    - ✅ Model versioning
    - ✅ Data validation
    - ✅ Separate train/test/holdout
    """)

with col2:
    st.markdown("""
    ### **Production Ready**
    - ✅ Health checks (/, /health, /ready)
    - ✅ CORS configuration
    - ✅ Environment-based config
    - ✅ Docker multi-stage builds
    - ✅ Non-root container user
    - ✅ AWS ECS optimized
    
    ### **Documentation**
    - ✅ API examples
    - ✅ Deployment guide
    - ✅ Code comments
    - ✅ This Streamlit app!
    - ✅ README files
    - ✅ Function docstrings
    """)

st.markdown("---")

# Next Steps
st.markdown(
    '<div class="section-header">🎯 Next Steps & Improvements</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### **Potential Enhancements**
    1. 📊 **A/B Testing Framework**
       - Compare model versions
       - Track business impact
    
    2. 🔄 **CI/CD Pipeline**
       - Automated testing
       - Continuous deployment
    
    3. 📈 **Monitoring & Alerting**
       - Prometheus metrics
       - Grafana dashboards
       - CloudWatch alarms
    
    4. 🎯 **Model Improvements**
       - Try other algorithms
       - Feature selection
       - Ensemble methods
    """)

with col2:
    st.markdown("""
    ### **Production Enhancements**
    1. 🔐 **Security**
       - API authentication (JWT)
       - Rate limiting
       - Input sanitization
    
    2. 💾 **Caching**
       - Redis for predictions
       - Feature store
    
    3. 📊 **Advanced Features**
       - SHAP explanations
       - Feature importance API
       - What-if analysis
    
    4. 🧪 **Testing**
       - Unit tests
       - Integration tests
       - Load testing
    """)

st.markdown("---")

# Conclusion
st.markdown("""
## 🎉 Conclusion

You now have a **complete end-to-end ML system**:

✅ **Data Processing** → Clean, validated data  
✅ **Feature Engineering** → 10 business-driven features  
✅ **Model Training** → Optimized XGBoost with MLflow  
✅ **Inference Engine** → Production-ready predictions  
✅ **REST API** → FastAPI with full endpoints  
✅ **Deployment** → Docker + AWS ECS ready  
✅ **Documentation** → Comprehensive Streamlit app  

### **Key Achievements**
- 📈 **95% Recall**: Catches almost all churners
- ⚡ **Fast API**: <100ms response time
- 🐳 **Containerized**: Deploy anywhere
- ☁️ **Cloud Ready**: AWS ECS configuration
- 📚 **Well Documented**: Every module explained

### **How to Use This Documentation**
1. **Learn**: Read through each section
2. **Experiment**: Try the code examples
3. **Deploy**: Follow deployment guides
4. **Extend**: Build on this foundation

---

### **Questions?**
- Check the **API Examples** in [API_EXAMPLES.md](API_EXAMPLES.md)
- Review **Deployment Steps** in [DEPLOYMENT.md](DEPLOYMENT.md)
- Test endpoints with **test_api.py**
- Explore notebooks in **notebooks/** folder

---

### **Project Repository**
📁 Location: `m:/local disk M/machine_learning/E2E_projects/sales-data-churn`

🎯 **Happy Predicting!** 🚀
""")

st.markdown("---")

# Footer
st.markdown(
    """
<div style="text-align: center; padding: 2rem; background-color: #F5F5F5; border-radius: 8px;">
    <h3>🎓 Churn Prediction ML Project</h3>
    <p>Complete End-to-End Machine Learning System</p>
    <p><strong>From Data to Deployment</strong></p>
    <p style="color: #666;">Built with ❤️ using Python, XGBoost, FastAPI, and AWS</p>
</div>
""",
    unsafe_allow_html=True,
)
