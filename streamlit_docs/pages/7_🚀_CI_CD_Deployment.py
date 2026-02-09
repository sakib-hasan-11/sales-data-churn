"""
Page 7: CI/CD Pipeline & Deployment
Complete testing and deployment automation documentation
"""

import streamlit as st

st.set_page_config(page_title="CI/CD & Deployment", page_icon="🚀", layout="wide")

# Custom CSS
st.markdown(
    """
<style>
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #D32F2F;
        margin-top: 2rem;
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #D32F2F;
    }
    .job-box {
        background-color: #E8F5E9;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #43A047;
    }
    .test-box {
        background-color: #E3F2FD;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #1E88E5;
    }
    .secret-box {
        background-color: #FFF3E0;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #FB8C00;
    }
    .job-name {
        font-family: 'Courier New', monospace;
        font-weight: bold;
        color: #2E7D32;
        font-size: 1.2rem;
    }
    .test-name {
        font-family: 'Courier New', monospace;
        font-weight: bold;
        color: #0D47A1;
        font-size: 1.1rem;
    }
    .code-inline {
        background-color: #F5F5F5;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
        color: #C62828;
    }
    .success-badge {
        background-color: #4CAF50;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.title("🚀 CI/CD Pipeline & Deployment")
st.markdown("### Automated Testing and AWS Deployment")
st.markdown("---")

# Overview
st.markdown("""
## 📋 Overview

This project includes a **complete CI/CD pipeline** that:
- 🧪 Runs **140+ automated tests** before deployment
- 🐳 Builds and tests **Docker containers**
- 🔒 Performs **security scanning**
- 🗄️ Verifies **model exists in S3** before deployment
- ☁️ Deploys to **Amazon ECR** automatically (API + Frontend)
- 📊 Monitors **performance metrics**

**Location**: `.github/workflows/ci-cd-pipeline.yml`

**Triggers**: Push to main/develop, Pull Requests

**AWS Resources**:
- **S3 Bucket**: `churn-project-model` (model storage)
- **ECR Repository 1**: `churn-prediction-api` (backend)
- **ECR Repository 2**: `churn-prediction-frontend` (frontend)

**Result**: Only deploys if ALL tests pass + model verified in S3! ✅
""")

st.markdown("---")

# Pipeline Overview
st.markdown(
    '<div class="section-header">🔄 Complete Pipeline Flow</div>',
    unsafe_allow_html=True,
)

st.markdown("""
### **13 Sequential Jobs** - All must pass before deployment!

```
📝 Push to GitHub
    ↓
1️⃣ Code Quality Checks (Black, Flake8, Isort)
    ↓
2️⃣ Data Processing Tests (25+ tests)
    ↓
3️⃣ Feature Engineering Tests (30+ tests)
    ↓
4️⃣ Inference Engine Tests (20+ tests)
    ↓
5️⃣ API Endpoint Tests (6+ tests)
    ↓
6️⃣ Edge Cases Tests (50+ tests)
    ↓
7️⃣ Performance Tests (10+ tests)
    ↓
8️⃣ Security Scanning (Bandit, Safety)
    ↓
9️⃣ Build & Test Docker Image
    ↓
🔟 Verify S3 Model Exists (main branch only)
    ↓
1️⃣1️⃣ Push API to Amazon ECR (main branch only)
    ↓
1️⃣2️⃣ Push Streamlit Frontend to ECR (main branch only)
    ↓
1️⃣3️⃣ Create Release (on version tags)
    ↓
✅ Deployed to Production!
```

**Total Pipeline Time**: ~15-18 minutes (full deployment)
**Success Rate Target**: 100% tests pass
**Artifacts**: 2 Docker images in ECR + Model in S3
""")

st.markdown("---")

# Job Details
st.markdown(
    '<div class="section-header">⚙️ Pipeline Jobs Breakdown</div>',
    unsafe_allow_html=True,
)

# Job 1
st.markdown('<div class="job-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="job-name">Job 1: Code Quality Checks</div>', unsafe_allow_html=True
)
st.markdown(
    """
**Purpose**: Ensure code follows best practices

**Tools**:
- **Black**: Python code formatter (PEP 8)
- **Isort**: Import statement organizer
- **Flake8**: Linting for code issues

**What it checks**:
```python
# Black - Formatting
✅ Consistent indentation
✅ Line length (88 characters)
✅ String quotes consistency

# Isort - Imports
✅ Sorted alphabetically
✅ Grouped by standard/third-party/local

# Flake8 - Linting
✅ Syntax errors
✅ Undefined variables
✅ Unused imports
```

**Example Issues Caught**:
- Inconsistent spacing
- Unused imports
- Long lines
- Missing docstrings
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Job 2
st.markdown('<div class="job-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="job-name">Job 2: Data Processing Tests</div>', unsafe_allow_html=True
)
st.markdown(
    """
**File**: `tests/test_data_processing.py`

**Modules Tested**:
1. `src/data_processing/load.py`
2. `src/data_processing/preprocess.py`
3. `src/utils/data_validator.py`

**Test Categories** (25+ tests):

**Load Data Tests**:
- ✅ Load CSV successfully
- ✅ All columns present
- ✅ Correct data types
- ✅ Handle missing files
- ✅ Handle corrupted files

**Preprocessing Tests**:
- ✅ Handle missing values
- ✅ Clean column names
- ✅ Remove outliers
- ✅ Preserve row count
- ✅ No NaN after processing

**Validation Tests**:
- ✅ Valid data passes
- ✅ Catch missing columns
- ✅ Catch wrong data types
- ✅ Handle empty DataFrames

**Integration Tests**:
- ✅ Complete pipeline works
- ✅ Target variable preserved
- ✅ Reproducible results
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Job 3
st.markdown('<div class="job-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="job-name">Job 3: Feature Engineering Tests</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
**File**: `tests/test_feature_engineering.py`

**Modules Tested**:
1. `src/features/build_feature.py`
2. `src/features/feature_preprocess.py`

**Test Categories** (30+ tests):

**Feature Creation Tests**:
- ✅ CLV calculated correctly
- ✅ Support efficiency created
- ✅ Payment reliability (0-1 range)
- ✅ Engagement score created
- ✅ Value to company metric
- ✅ Tenure categories binned
- ✅ Age groups created
- ✅ Spend categories created
- ✅ No infinite values
- ✅ No NaN introduced

**Feature Preprocessing Tests**:
- ✅ Returns (X, y, feature_names)
- ✅ Correct shapes
- ✅ Gender encoded (0/1)
- ✅ Categoricals one-hot encoded
- ✅ Original categoricals removed
- ✅ All features numeric
- ✅ No missing values in X or y

**Integration Tests**:
- ✅ Full pipeline works
- ✅ Produces ML-ready data
- ✅ Reproducible results
- ✅ Target is binary (0/1)
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Job 4
st.markdown('<div class="job-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="job-name">Job 4: Inference Engine Tests</div>', unsafe_allow_html=True
)
st.markdown(
    """
**File**: `tests/test_inference.py`

**Module Tested**: `src/inference/inference.py`

**Test Categories** (20+ tests):

**InferencePreprocessor Tests**:
- ✅ Clean column names
- ✅ Encode features correctly
- ✅ Align features to training
- ✅ Handle dict input
- ✅ Handle DataFrame input

**ModelLoader Tests**:
- ✅ Load from .pkl file
- ✅ Load from MLflow run
- ✅ Load best from experiment
- ✅ Handle missing files
- ✅ Extract feature names

**ChurnPredictor Tests**:
- ✅ Load model successfully
- ✅ Single prediction works
- ✅ Batch prediction works
- ✅ Risk level calculation (Low/Medium/High/Critical)
- ✅ Return correct format
- ✅ Probability in 0-1 range
- ✅ Prediction in {0, 1}
- ✅ Get model info

**Integration Tests**:
- ✅ End-to-end prediction works
- ✅ Handles multiple customers
- ✅ Consistent results
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Job 5
st.markdown('<div class="job-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="job-name">Job 5: API Endpoint Tests</div>', unsafe_allow_html=True
)
st.markdown(
    """
**File**: `test_api.py`

**API Tested**: `main.py` (FastAPI application)

**Test Categories** (6+ tests):

**Endpoint Tests**:

1. **GET /** - Root endpoint
   ```python
   ✅ Returns welcome message
   ✅ Status 200
   ✅ Contains "Churn Prediction API"
   ```

2. **GET /health** - Health check
   ```python
   ✅ Returns {"status": "healthy"}
   ✅ Model loaded = true
   ✅ Status 200
   ```

3. **GET /ready** - Readiness probe
   ```python
   ✅ Returns model info
   ✅ Shows feature count
   ✅ Shows experiment name
   ```

4. **POST /predict** - Single prediction
   ```python
   ✅ Accepts customer data
   ✅ Returns churn probability
   ✅ Returns prediction (0/1)
   ✅ Returns risk level
   ✅ Validates input with Pydantic
   ```

5. **POST /predict/batch** - Batch prediction
   ```python
   ✅ Accepts list of customers
   ✅ Returns all predictions
   ✅ Returns aggregate stats
   ✅ Shows high risk count
   ```

6. **GET /model/info** - Model metadata
   ```python
   ✅ Returns model source
   ✅ Shows run ID
   ✅ Shows feature count
   ```
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Job 6
st.markdown('<div class="job-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="job-name">Job 6: Edge Cases Tests</div>', unsafe_allow_html=True
)
st.markdown(
    """
**File**: `tests/test_edge_cases.py`

**Purpose**: Test error handling and edge cases

**Test Categories** (50+ tests):

**Missing Values**:
- ✅ Missing single field
- ✅ Multiple missing fields
- ✅ None values
- ✅ Empty strings

**Invalid Data Types**:
- ✅ String for numeric field
- ✅ Negative ages
- ✅ Negative tenure
- ✅ Non-numeric prices

**Out of Range Values**:
- ✅ Age = 200
- ✅ Spend = 9,999,999
- ✅ Tenure = 0
- ✅ Extreme values

**Empty Input**:
- ✅ Empty dict
- ✅ Empty list
- ✅ Null payload

**Malformed Input**:
- ✅ Invalid gender
- ✅ Unknown subscription type
- ✅ Invalid contract length

**Special Characters**:
- ✅ Customer ID with @#$%
- ✅ Special chars in strings

**Boundary Values**:
- ✅ Minimum age (18)
- ✅ Maximum values
- ✅ All zeros
- ✅ All nulls

**Large Batches**:
- ✅ 1,000 customers
- ✅ 5,000 customers
- ✅ Memory efficiency

**Duplicates**:
- ✅ Duplicate customer IDs
- ✅ Identical data
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Job 7
st.markdown('<div class="job-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="job-name">Job 7: Performance Tests</div>', unsafe_allow_html=True
)
st.markdown(
    """
**File**: `tests/test_performance.py`

**Purpose**: Ensure API meets performance requirements

**Test Categories** (10+ tests):

**Latency Tests**:
```python
✅ Single prediction < 100ms
✅ Average latency tracked
✅ P95 latency < 150ms
✅ P99 latency < 200ms
```

**Throughput Tests**:
```python
✅ Batch faster than sequential
✅ 100 predictions throughput
✅ 1,000 predictions throughput
✅ Target: >10 predictions/sec
```

**Memory Tests**:
```python
✅ Large batches (5,000) don't OOM
✅ Memory usage reasonable
✅ No memory leaks
```

**Stress Tests**:
```python
✅ 1,000 sequential predictions
✅ Sustained load handling
✅ No degradation over time
```

**Concurrency Tests**:
```python
✅ Multiple simultaneous requests
✅ Thread safety
✅ No race conditions
```

**Benchmarks**:
- Single prediction: ~20-50ms
- Batch (100): ~500ms = 200 pred/sec
- Batch (1000): ~3-5sec = 200-300 pred/sec
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Job 8
st.markdown('<div class="job-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="job-name">Job 8: Security Scanning</div>', unsafe_allow_html=True
)
st.markdown(
    """
**Purpose**: Identify security vulnerabilities

**Tools**:

1. **Safety** - Dependency vulnerabilities
   ```python
   ✅ Checks requirements.txt
   ✅ Known CVEs in packages
   ✅ Outdated packages
   ✅ Security advisories
   ```

2. **Bandit** - Code security analysis
   ```python
   ✅ Hardcoded secrets
   ✅ SQL injection risks
   ✅ Insecure random usage
   ✅ Weak crypto
   ✅ Assert statements in production
   ```

**Output**: Security report uploaded as artifact

**Action**: Pipeline fails if critical vulnerabilities found
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Job 9
st.markdown('<div class="job-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="job-name">Job 9: Build & Test Docker Image</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
**Purpose**: Build and verify Docker container

**Steps**:

1. **Build Image**
   ```bash
   docker build -t churn-api:test .
   ```
   - Multi-stage build (builder + production)
   - Python 3.11 slim base
   - Non-root user
   - Optimized layers

2. **Start Container**
   ```bash
   docker run -d -p 8000:8000 --name test-api churn-api:test
   ```

3. **Test Health Checks**
   ```bash
   ✅ curl http://localhost:8000/health
   ✅ curl http://localhost:8000/ready
   ```

4. **Test Prediction Endpoint**
   ```bash
   ✅ POST /predict with test data
   ✅ Verify response format
   ✅ Check latency
   ```

5. **Security Scan (Trivy)**
   ```bash
   ✅ Scan for CVEs
   ✅ Check base image vulnerabilities
   ✅ Check installed packages
   ```

6. **Verify Container Logs**
   ```bash
   ✅ No errors on startup
   ✅ Model loaded successfully
   ✅ API listening on port 8000
   ```

**Success Criteria**:
- Image builds without errors
- Container starts and runs
- All health checks pass
- No critical CVEs
- Image size < 2GB
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Job 10
st.markdown('<div class="job-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="job-name">Job 10: Verify S3 Model</div>', unsafe_allow_html=True
)
st.markdown(
    """
**Purpose**: Ensure trained model exists in S3 before deployment

**Trigger**: Only on `main` branch push + all tests passed

**Steps**:

1. **Configure AWS Credentials**
   ```yaml
   - uses: aws-actions/configure-aws-credentials@v4
     with:
       aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
       aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
       aws-region: us-east-1
   ```

2. **Check Model Exists in S3**
   ```bash
   # List model file
   aws s3 ls s3://churn-project-model/models/churn_model_production.pkl
   
   # Output if exists:
   # 2024-02-08 10:30:45   15234567 churn_model_production.pkl
   ```

3. **Get Model Metadata**
   ```bash
   aws s3api head-object \\
     --bucket churn-project-model \\
     --key models/churn_model_production.pkl \\
     --query '{Size:ContentLength,LastModified:LastModified,Metadata:Metadata}'
   ```

**S3 Configuration**:
- **Bucket**: `churn-project-model`
- **Path**: `models/churn_model_production.pkl`
- **Region**: `us-east-1`
- **Access**: Private (IAM credentials required)

**What it Validates**:
✅ Model file exists in S3
✅ File is accessible with current credentials
✅ Model was trained and uploaded successfully
✅ Ready for container deployment

**Success**: Model verified → Proceed to ECR push
**Failure**: Model missing → Stop pipeline (prevents deploying without model)

**Why This Matters**:
- API containers load model from S3 at startup
- No model = API fails to start
- Prevents deploying broken containers
- Ensures model-code version compatibility
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Job 11
st.markdown('<div class="job-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="job-name">Job 11: Push API to Amazon ECR</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
**Purpose**: Build and push API Docker image to Amazon ECR

**Trigger**: After S3 model verification passes (main branch only)

**Repository**: `churn-prediction-api`

**Steps**:

1. **Configure AWS Credentials**
   ```bash
   # Same credentials as S3 verification
   AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
   AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
   AWS_REGION: us-east-1
   ```

2. **Login to Amazon ECR**
   ```bash
   aws ecr get-login-password --region us-east-1 | \\
     docker login --username AWS --password-stdin \\
     <account-id>.dkr.ecr.us-east-1.amazonaws.com
   ```

3. **Set Up Docker Buildx**
   ```bash
   # Multi-platform build support
   docker buildx create --use
   ```

4. **Extract Docker Metadata & Tags**
   ```yaml
   tags: |
     type=ref,event=branch          # branch name
     type=ref,event=pr               # PR number
     type=semver,pattern={{version}} # v1.2.3
     type=semver,pattern={{major}}.{{minor}}  # v1.2
     type=sha,prefix={{branch}}-     # main-abc123
     type=raw,value=latest,enable={{is_default_branch}}  # latest
   ```
   
   **Resulting Tags**:
   - `churn-prediction-api:latest`
   - `churn-prediction-api:main`
   - `churn-prediction-api:main-abc123def` (commit SHA)

5. **Build and Push Docker Image**
   ```bash
   docker build -t <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-api:latest .
   docker push <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-api:latest
   ```
   
   **Build Arguments Passed**:
   ```dockerfile
   BUILD_DATE=${{ github.event.head_commit.timestamp }}
   VCS_REF=${{ github.sha }}
   MODEL_SOURCE=s3
   S3_BUCKET_NAME=churn-project-model
   S3_MODEL_NAME=churn_model_production.pkl
   ```

6. **Verify Image in ECR**
   ```bash
   aws ecr describe-images \\
     --repository-name churn-prediction-api \\
     --image-ids imageTag=latest \\
     --region us-east-1
   ```
   
   **Output**:
   ```json
   {
     "imageDetails": [{
       "imageTag": "latest",
       "imageSizeInBytes": 1234567890,
       "imagePushedAt": "2024-02-08T10:45:30Z",
       "imageDigest": "sha256:abc123..."
     }]
   }
   ```

7. **Print Deployment Instructions**
   ```bash
   echo "✅ Successfully pushed to ECR"
   echo "Image: <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-api:latest"
   echo "Model Source: S3 bucket churn-project-model"
   echo "Commit: ${{ github.sha }}"
   echo ""
   echo "To pull this image:"
   echo "  aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com"
   echo "  docker pull <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-api:latest"
   ```

**Image Details**:
- **Base Image**: `python:3.11-slim`
- **Size**: ~800MB (optimized multi-stage build)
- **Model Loading**: From S3 at container startup
- **Environment Variables**:
  - `MODEL_SOURCE=s3`
  - `S3_BUCKET_NAME=churn-project-model`
  - `S3_MODEL_NAME=churn_model_production.pkl`
  - `AWS_REGION=us-east-1`

**Cache Optimization**:
```yaml
cache-from: type=gha        # Use GitHub Actions cache
cache-to: type=gha,mode=max # Save layers for next build
```
- First build: ~8-10 minutes
- Cached builds: ~2-3 minutes

**Success Criteria**:
✅ Image builds successfully
✅ Pushes to ECR without errors
✅ Tagged with commit SHA for traceability
✅ Available for deployment

**Result**: 
API container ready in ECR → Can be deployed to ECS/Fargate/EC2
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Job 12
st.markdown('<div class="job-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="job-name">Job 12: Push Streamlit Frontend to ECR</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
**Purpose**: Build and push Streamlit frontend Docker image to Amazon ECR

**Trigger**: After API push succeeds (main branch only)

**Repository**: `churn-prediction-frontend`

**Steps**:

1. **Configure AWS & Login to ECR**
   ```bash
   # Same authentication as API push
   aws ecr get-login-password --region us-east-1 | \\
     docker login --username AWS --password-stdin \\
     <account-id>.dkr.ecr.us-east-1.amazonaws.com
   ```

2. **Set Up Docker Buildx**
   ```bash
   docker buildx create --use
   ```

3. **Extract Docker Metadata & Tags**
   ```yaml
   tags: |
     type=ref,event=branch          # main
     type=ref,event=pr               # PR number
     type=semver,pattern={{version}} # v1.2.3
     type=semver,pattern={{major}}.{{minor}}
     type=sha,prefix={{branch}}-     # main-abc123
     type=raw,value=latest,enable={{is_default_branch}}  # latest
   ```
   
   **Resulting Tags**:
   - `churn-prediction-frontend:latest`
   - `churn-prediction-frontend:main`
   - `churn-prediction-frontend:main-abc123def`

4. **Build and Push Streamlit Docker Image**
   ```bash
   docker build \\
     -f Dockerfile.streamlit \\
     -t <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-frontend:latest .
   
   docker push <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-frontend:latest
   ```
   
   **Build Arguments**:
   ```dockerfile
   BUILD_DATE=${{ github.event.head_commit.timestamp }}
   VCS_REF=${{ github.sha }}
   ```

5. **Verify Streamlit Image in ECR**
   ```bash
   aws ecr describe-images \\
     --repository-name churn-prediction-frontend \\
     --image-ids imageTag=latest \\
     --region us-east-1
   ```

6. **Print Deployment Instructions**
   ```bash
   echo "✅ Streamlit Frontend image pushed to ECR successfully"
   echo "Image: <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-frontend:latest"
   echo "Commit: ${{ github.sha }}"
   echo ""
   echo "To pull this image:"
   echo "  aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com"
   echo "  docker pull <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-frontend:latest"
   echo ""
   echo "To run the Streamlit frontend:"
   echo "  docker run -d -p 8501:8501 -e API_URL=http://your-api-url:8000 <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-frontend:latest"
   ```

**Streamlit Image Details**:
- **Base Image**: `python:3.11-slim`
- **Size**: ~1.2GB (includes Streamlit + visualization libraries)
- **Exposed Port**: `8501` (Streamlit default)
- **Environment Variables**:
  - `API_URL` - Backend API endpoint URL
  - `STREAMLIT_SERVER_PORT=8501`
  - `STREAMLIT_SERVER_ADDRESS=0.0.0.0`

**What's Included**:
- Streamlit application (`streamlit_app.py`)
- Documentation pages (`streamlit_docs/`)
- CSS styling and assets
- Requirements (`requirements_streamlit.txt`)

**Cache Optimization**:
```yaml
cache-from: type=gha
cache-to: type=gha,mode=max
```
- First build: ~6-8 minutes
- Cached builds: ~1-2 minutes

**Success Criteria**:
✅ Streamlit image builds successfully
✅ Pushes to ECR without errors
✅ Tagged with commit SHA
✅ Ready for deployment

**Deployment Configuration**:
```bash
# Environment variable needed at runtime
docker run -d \\
  -p 8501:8501 \\
  -e API_URL=http://api-service:8000 \\
  --name streamlit-frontend \\
  <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-frontend:latest
```

**Result**: 
Frontend container ready in ECR → Can be deployed alongside API

**Complete Stack**:
- Backend API: Port 8000 (churn-prediction-api)
- Frontend: Port 8501 (churn-prediction-frontend)
- Model: Loaded from S3 (churn-project-model)
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# S3 Bucket Details Section
st.markdown(
    '<div class="section-header">☁️ S3 Model Storage Details</div>',
    unsafe_allow_html=True,
)

st.markdown("""
### **S3 Bucket Configuration**

The trained model is stored in Amazon S3 for centralized access:

**Bucket Structure**:
```
s3://churn-project-model/
├── models/
│   ├── churn_model_production.pkl     # Production model
│   ├── churn_model_v1.0.0.pkl        # Versioned backups
│   └── model_metadata.json            # Model info
├── artifacts/
│   ├── feature_names.json
│   └── preprocessing_config.json
└── experiments/
    └── mlflow_runs/
```

**Production Model Details**:
- **Bucket**: `churn-project-model`
- **Key**: `models/churn_model_production.pkl`
- **Region**: `us-east-1`
- **Size**: ~15-20 MB (pickled scikit-learn model)
- **Format**: Pickle (.pkl)
- **Access**: Private (IAM credentials required)

**IAM Permissions Required**:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::churn-project-model/*",
        "arn:aws:s3:::churn-project-model"
      ]
    }
  ]
}
```

**Upload Model to S3** (one-time setup):
```bash
# Using AWS CLI
aws s3 cp models/churn_model_production.pkl \\
  s3://churn-project-model/models/churn_model_production.pkl \\
  --region us-east-1

# Verify upload
aws s3 ls s3://churn-project-model/models/

# Get model info
aws s3api head-object \\
  --bucket churn-project-model \\
  --key models/churn_model_production.pkl
```

**Using Python (boto3)**:
```python
import boto3

# Initialize S3 client
s3_client = boto3.client('s3', region_name='us-east-1')

# Upload model
with open('models/churn_model_production.pkl', 'rb') as f:
    s3_client.put_object(
        Bucket='churn-project-model',
        Key='models/churn_model_production.pkl',
        Body=f
    )

print("✅ Model uploaded to S3")
```

**Model Versioning Strategy**:
```bash
# Keep production model
s3://churn-project-model/models/churn_model_production.pkl

# Archive versions with timestamp
s3://churn-project-model/models/archive/churn_model_2024-02-08_v1.pkl
s3://churn-project-model/models/archive/churn_model_2024-02-07_v1.pkl

# Rollback if needed
aws s3 cp \\
  s3://churn-project-model/models/archive/churn_model_2024-02-07_v1.pkl \\
  s3://churn-project-model/models/churn_model_production.pkl
```

**Benefits of S3 Storage**:
- ✅ **Centralized**: Single source of truth
- ✅ **Versioned**: Keep model history
- ✅ **Scalable**: No container size limits
- ✅ **Accessible**: Multiple containers can load same model
- ✅ **Durable**: 99.999999999% durability (11 nines)
- ✅ **Secure**: IAM-based access control

**Container Integration**:
```dockerfile
# Dockerfile
ENV MODEL_SOURCE=s3
ENV S3_BUCKET_NAME=churn-project-model
ENV S3_MODEL_NAME=churn_model_production.pkl
ENV AWS_REGION=us-east-1

# Model loaded at container startup via inference.py
```

**Monitoring S3 Usage**:
```bash
# Check model access logs
aws s3api get-bucket-logging --bucket churn-project-model

# Monitor S3 requests in CloudWatch
aws cloudwatch get-metric-statistics \\
  --namespace AWS/S3 \\
  --metric-name NumberOfObjects \\
  --dimensions Name=BucketName,Value=churn-project-model
```
""")

st.markdown("---")

# ECR Details Section
st.markdown(
    '<div class="section-header">🐳 Amazon ECR Repository Details</div>',
    unsafe_allow_html=True,
)

st.markdown("""
### **ECR Repository Configuration**

Two repositories are created for API and Frontend:

**Repository 1: API Backend**
- **Name**: `churn-prediction-api`
- **Region**: `us-east-1`
- **URI**: `<account-id>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-api`
- **Tag Strategy**: `latest`, `main`, `main-<commit-sha>`

**Repository 2: Streamlit Frontend**
- **Name**: `churn-prediction-frontend`
- **Region**: `us-east-1`
- **URI**: `<account-id>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-frontend`
- **Tag Strategy**: `latest`, `main`, `main-<commit-sha>`

**Create ECR Repositories** (one-time setup):
```bash
# Create API repository
aws ecr create-repository \\
  --repository-name churn-prediction-api \\
  --region us-east-1 \\
  --image-scanning-configuration scanOnPush=true \\
  --encryption-configuration encryptionType=AES256

# Create Frontend repository
aws ecr create-repository \\
  --repository-name churn-prediction-frontend \\
  --region us-east-1 \\
  --image-scanning-configuration scanOnPush=true \\
  --encryption-configuration encryptionType=AES256
```

**List Images in ECR**:
```bash
# List API images
aws ecr list-images \\
  --repository-name churn-prediction-api \\
  --region us-east-1

# List Frontend images
aws ecr list-images \\
  --repository-name churn-prediction-frontend \\
  --region us-east-1
```

**Pull Images Locally**:
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \\
  docker login --username AWS --password-stdin \\
  <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Pull API image
docker pull <account-id>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-api:latest

# Pull Frontend image
docker pull <account-id>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-frontend:latest
```

**Image Lifecycle Policy** (auto-cleanup):
```json
{
  "rules": [
    {
      "rulePriority": 1,
      "description": "Keep last 10 images",
      "selection": {
        "tagStatus": "any",
        "countType": "imageCountMoreThan",
        "countNumber": 10
      },
      "action": {
        "type": "expire"
      }
    }
  ]
}
```

**Apply Lifecycle Policy**:
```bash
aws ecr put-lifecycle-policy \\
  --repository-name churn-prediction-api \\
  --lifecycle-policy-text file://lifecycle-policy.json
```

**Security Features**:
- ✅ **Image Scanning**: Automatic vulnerability scanning on push
- ✅ **Encryption**: AES256 encryption at rest
- ✅ **IAM Access**: Fine-grained access control
- ✅ **Private**: Not publicly accessible
- ✅ **Audit Logs**: CloudTrail logging

**Cost Optimization**:
```bash
# View repository size
aws ecr describe-repositories \\
  --repository-names churn-prediction-api churn-prediction-frontend

# Storage cost: $0.10 per GB/month
# Transfer: Free within same region
```

**CI/CD Integration**:
```yaml
# Automatic push on main branch
- Push code to main
- Run all tests
- Verify S3 model
- Build Docker images
- Push to ECR (if tests pass)
- Deploy to ECS (optional)
```
""")

st.markdown("---")

# GitHub Secrets
st.markdown(
    '<div class="section-header">🔐 GitHub Secrets Configuration</div>',
    unsafe_allow_html=True,
)

st.markdown("""
### **Required Secrets**

To enable ECR deployment, you must configure these secrets in your GitHub repository:
""")

st.markdown('<div class="secret-box">', unsafe_allow_html=True)
st.markdown('<div class="test-name">🔑 AWS_ACCESS_KEY_ID</div>', unsafe_allow_html=True)
st.markdown(
    """
**Description**: Your AWS IAM user access key ID

**Format**: String (e.g., `AKIAIOSFODNN7EXAMPLE`)

**How to get it**:
```bash
# 1. Create IAM user
aws iam create-user --user-name github-actions-ecr

# 2. Attach ECR policy
aws iam attach-user-policy \\
  --user-name github-actions-ecr \\
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser

# 3. Create access key
aws iam create-access-key --user-name github-actions-ecr
# Save the AccessKeyId!
```

**Add to GitHub**:
1. Go to: `Settings → Secrets and variables → Actions`
2. Click `New repository secret`
3. Name: `AWS_ACCESS_KEY_ID`
4. Value: Your access key ID
5. Click `Add secret`
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="secret-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="test-name">🔑 AWS_SECRET_ACCESS_KEY</div>', unsafe_allow_html=True
)
st.markdown(
    """
**Description**: Your AWS IAM user secret access key

**Format**: String (e.g., `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`)

**How to get it**:
- Same as above - created with the access key
- ⚠️ **CRITICAL**: Save immediately! You can't retrieve it later!

**Add to GitHub**:
1. Same process as AWS_ACCESS_KEY_ID
2. Name: `AWS_SECRET_ACCESS_KEY`
3. Value: Your secret access key
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="secret-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="test-name">🔑 ECS_CLUSTER (Optional)</div>', unsafe_allow_html=True
)
st.markdown(
    """
**Description**: Name of your ECS cluster (for automatic deployment)

**Format**: String (e.g., `churn-prediction-cluster`)

**When needed**: Only if you want automatic ECS service updates

**How to get it**:
1. Go to AWS ECS Console
2. Find your cluster
3. Copy the cluster name
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="secret-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="test-name">🔑 ECS_SERVICE (Optional)</div>', unsafe_allow_html=True
)
st.markdown(
    """
**Description**: Name of your ECS service

**Format**: String (e.g., `churn-api-service`)

**When needed**: Only if you want automatic ECS service updates

**How to get it**:
1. Go to your ECS cluster
2. Find your service
3. Copy the service name
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Deployment Process
st.markdown(
    '<div class="section-header">☁️ Deployment to AWS (S3 + ECR)</div>',
    unsafe_allow_html=True,
)

st.markdown("""
### **Automatic Deployment** (Main Branch Only)

When you push to the `main` branch and all tests pass:
""")

st.markdown('<div class="job-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="job-name">Deployment Flow: S3 + ECR</div>', unsafe_allow_html=True
)
st.markdown(
    """
**Trigger**: Push to `main` branch + all 140+ tests passed

**Complete Deployment Process**:

**Stage 1: Verify Model in S3** (Job 10)
```bash
# Check model exists and is accessible
aws s3 ls s3://churn-project-model/models/churn_model_production.pkl

✅ Model verified
✅ Ready for container deployment
```

**Stage 2: Push API to ECR** (Job 11)
```bash
# Authenticate to AWS ECR
aws ecr get-login-password --region us-east-1 | \\
  docker login --username AWS --password-stdin \\
  <account>.dkr.ecr.us-east-1.amazonaws.com

# Build Docker image
docker build -t churn-prediction-api:latest .

# Tag for ECR
docker tag churn-prediction-api:latest \\
  <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-api:latest

docker tag churn-prediction-api:latest \\
  <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-api:main-<commit-sha>

# Push to ECR
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-api:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-api:main-<commit-sha>

✅ API image pushed to ECR
✅ Tagged with: latest, main, main-<commit-sha>
```

**Stage 3: Push Streamlit Frontend to ECR** (Job 12)
```bash
# Build Streamlit Docker image
docker build -f Dockerfile.streamlit -t churn-prediction-frontend:latest .

# Tag for ECR
docker tag churn-prediction-frontend:latest \\
  <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-frontend:latest

docker tag churn-prediction-frontend:latest \\
  <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-frontend:main-<commit-sha>

# Push to ECR
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-frontend:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-frontend:main-<commit-sha>

✅ Frontend image pushed to ECR
✅ Tagged with: latest, main, main-<commit-sha>
```

**Verification**:
```bash
# Verify both images in ECR
aws ecr describe-images \\
  --repository-name churn-prediction-api \\
  --image-ids imageTag=latest

aws ecr describe-images \\
  --repository-name churn-prediction-frontend \\
  --image-ids imageTag=latest
```

**Result**: 
- ✅ Model verified in S3: `s3://churn-project-model/models/churn_model_production.pkl`
- ✅ API image in ECR: `churn-prediction-api:latest`
- ✅ Frontend image in ECR: `churn-prediction-frontend:latest`
- ✅ Both tagged with commit SHA for traceability
- ✅ Ready for deployment to ECS/Fargate/EC2
- ✅ Zero downtime deployment possible

**Deployment Commands** (after push):
```bash
# Pull and run API
docker pull <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-api:latest
docker run -d -p 8000:8000 \\
  -e MODEL_SOURCE=s3 \\
  -e S3_BUCKET_NAME=churn-project-model \\
  -e S3_MODEL_NAME=churn_model_production.pkl \\
  -e AWS_REGION=us-east-1 \\
  --name churn-api \\
  <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-api:latest

# Pull and run Frontend
docker pull <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-frontend:latest
docker run -d -p 8501:8501 \\
  -e API_URL=http://<api-host>:8000 \\
  --name churn-frontend \\
  <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-frontend:latest
```
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Complete Workflow Example
st.markdown(
    '<div class="section-header">📝 Complete Deployment Example</div>',
    unsafe_allow_html=True,
)

st.markdown("""
### **Scenario**: Deploy a new feature

**Step 1: Make Changes**
```bash
# Create feature branch
git checkout -b feature/improve-predictions

# Make changes to code
# ... edit files ...

# Commit changes
git add .
git commit -m "Improve prediction accuracy"

# Push to GitHub
git push origin feature/improve-predictions
```

**What Happens**:
- ✅ GitHub Actions triggers
- ✅ Runs ALL tests (no deployment)
- ✅ Reports status on PR

---

**Step 2: Create Pull Request**
```
GitHub UI → Create Pull Request
```

**What Happens**:
- ✅ CI runs again on PR
- ✅ Shows test results in PR
- ✅ Blocks merge if tests fail
- ✅ Team can review changes

---

**Step 3: Merge to Main**
```bash
# After PR approval
git checkout main
git pull origin main
git merge feature/improve-predictions
git push origin main
```

**What Happens** (Full Pipeline):
```
1. Code Quality ✅ (30 seconds)
2. Data Processing Tests ✅ (2 minutes)
3. Feature Engineering Tests ✅ (2 minutes)
4. Inference Tests ✅ (1 minute)
5. API Tests ✅ (1 minute)
6. Edge Cases Tests ✅ (2 minutes)
7. Performance Tests ✅ (1 minute)
8. Security Scan ✅ (1 minute)
9. Build & Test Docker ✅ (3 minutes)
10. Verify S3 Model ✅ (30 seconds)
11. Push API to ECR ✅ (2 minutes)
12. Push Frontend to ECR ✅ (2 minutes)

Total: ~15-18 minutes
```

**Result**:
- 🎉 New version deployed to AWS!
- 📦 Model verified in S3: `s3://churn-project-model/models/churn_model_production.pkl`
- 🐳 API image in ECR: `churn-prediction-api:latest`
- 🎨 Frontend image in ECR: `churn-prediction-frontend:latest`
- 🚀 Ready for ECS/Fargate deployment
- 🔍 Fully tested and validated

---

**Step 4: Verify Deployment**
```bash
# Check S3 model
aws s3 ls s3://churn-project-model/models/
aws s3api head-object \\
  --bucket churn-project-model \\
  --key models/churn_model_production.pkl

# Check ECR images
aws ecr list-images \\
  --repository-name churn-prediction-api \\
  --region us-east-1

aws ecr list-images \\
  --repository-name churn-prediction-frontend \\
  --region us-east-1

# Get image details
aws ecr describe-images \\
  --repository-name churn-prediction-api \\
  --image-ids imageTag=latest

aws ecr describe-images \\
  --repository-name churn-prediction-frontend \\
  --image-ids imageTag=latest

# Test locally (optional)
aws ecr get-login-password --region us-east-1 | \\
  docker login --username AWS --password-stdin \\
  <account>.dkr.ecr.us-east-1.amazonaws.com

docker pull <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-api:latest
docker run -d -p 8000:8000 \\
  -e MODEL_SOURCE=s3 \\
  -e S3_BUCKET_NAME=churn-project-model \\
  -e S3_MODEL_NAME=churn_model_production.pkl \\
  <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-api:latest

# Test API
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -d '{...}'
```

---

**Step 5: Deploy to Production (ECS/Fargate)**
```bash
# Option 1: Update ECS service (if using ECS)
aws ecs update-service \\
  --cluster churn-prediction-cluster \\
  --service churn-api-service \\
  --force-new-deployment

aws ecs update-service \\
  --cluster churn-prediction-cluster \\
  --service churn-frontend-service \\
  --force-new-deployment

# Option 2: Use docker-compose (for EC2)
# Create docker-compose.yml with ECR images
docker-compose pull
docker-compose up -d

# Option 3: Manual docker run
docker run -d -p 8000:8000 \\
  --name churn-api \\
  -e MODEL_SOURCE=s3 \\
  -e S3_BUCKET_NAME=churn-project-model \\
  -e S3_MODEL_NAME=churn_model_production.pkl \\
  <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-api:latest

docker run -d -p 8501:8501 \\
  --name churn-frontend \\
  -e API_URL=http://api-host:8000 \\
  <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-frontend:latest
```
""")

st.markdown("---")

# Test Statistics
st.markdown(
    '<div class="section-header">📊 Test Coverage Statistics</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Tests", "140+", help="Across all test suites")
with col2:
    st.metric("Test Files", "6", help="Separate test modules")
with col3:
    st.metric("Pipeline Time", "~15 min", help="Full pipeline duration")
with col4:
    st.metric("Success Rate", "100%", help="Required to deploy")

st.markdown("---")

# Test Breakdown Table
st.markdown("### **Test Breakdown by Category**")

test_data = {
    "Category": [
        "Data Processing",
        "Feature Engineering",
        "Inference Engine",
        "API Endpoints",
        "Edge Cases",
        "Performance",
        "Security",
        "Docker",
        "**TOTAL**",
    ],
    "Test File": [
        "test_data_processing.py",
        "test_feature_engineering.py",
        "test_inference.py",
        "test_api.py",
        "test_edge_cases.py",
        "test_performance.py",
        "Bandit, Safety",
        "Container tests",
        "**All files**",
    ],
    "Test Count": ["25+", "30+", "20+", "6+", "50+", "10+", "Auto", "5+", "**140+**"],
    "What's Tested": [
        "Load, clean, validate data",
        "10 features, encoding, scaling",
        "Preprocessing, predictions, risk",
        "All REST endpoints",
        "Missing values, boundaries, errors",
        "Latency <100ms, throughput",
        "CVEs, code vulnerabilities",
        "Build, health, endpoints",
        "**Complete coverage**",
    ],
}

st.table(test_data)

st.markdown("---")

# Prerequisites
st.markdown(
    '<div class="section-header">✅ Setup Prerequisites</div>', unsafe_allow_html=True
)

st.markdown("""
### **Before Running the Pipeline**

**1. Create S3 Bucket for Model Storage**
```bash
# Create S3 bucket
aws s3 mb s3://churn-project-model --region us-east-1

# Enable versioning (recommended)
aws s3api put-bucket-versioning \\
  --bucket churn-project-model \\
  --versioning-configuration Status=Enabled

# Upload trained model
aws s3 cp models/churn_model_production.pkl \\
  s3://churn-project-model/models/churn_model_production.pkl

# Verify upload
aws s3 ls s3://churn-project-model/models/
```

**2. Create ECR Repositories**
```bash
# Create API repository
aws ecr create-repository \\
  --repository-name churn-prediction-api \\
  --region us-east-1 \\
  --image-scanning-configuration scanOnPush=true

# Create Frontend repository
aws ecr create-repository \\
  --repository-name churn-prediction-frontend \\
  --region us-east-1 \\
  --image-scanning-configuration scanOnPush=true
```

**3. Create IAM User with Required Permissions**
```bash
# Create user
aws iam create-user --user-name github-actions-deploy

# Create policy file (deploy-policy.json):
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:PutImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:DescribeRepositories",
        "ecr:DescribeImages"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket",
        "s3:HeadObject"
      ],
      "Resource": [
        "arn:aws:s3:::churn-project-model",
        "arn:aws:s3:::churn-project-model/*"
      ]
    }
  ]
}

# Create and attach policy
aws iam create-policy \\
  --policy-name GitHubActionsDeployPolicy \\
  --policy-document file://deploy-policy.json

aws iam attach-user-policy \\
  --user-name github-actions-deploy \\
  --policy-arn arn:aws:iam::<account-id>:policy/GitHubActionsDeployPolicy

# Create access key
aws iam create-access-key --user-name github-actions-deploy
# ⚠️ Save the AccessKeyId and SecretAccessKey!
```

**4. Add GitHub Secrets**
- Go to: `Repository → Settings → Secrets and variables → Actions`
- Click `New repository secret`
- Add the following secrets:
  - `AWS_ACCESS_KEY_ID`: Your IAM access key ID
  - `AWS_SECRET_ACCESS_KEY`: Your IAM secret access key

**5. Create Test Model**
```bash
# Required for CI/CD tests
python tests/create_test_model.py

# Verify test model created
ls -lh tests/test_model.pkl
```

**6. Commit All Files**
```bash
git add .github/workflows/ci-cd-pipeline.yml
git add tests/
git add Dockerfile
git add Dockerfile.streamlit
git add docker-compose.yml
git commit -m "Add CI/CD pipeline with S3 and ECR deployment"
git push origin main
```

**7. Monitor First Run**
- Go to: `GitHub → Actions tab`
- Watch pipeline execute
- Verify all jobs pass
- Check AWS:
  - S3 model verified ✅
  - ECR API image pushed ✅
  - ECR Frontend image pushed ✅

**8. Verify AWS Resources**
```bash
# Check S3 model
aws s3 ls s3://churn-project-model/models/

# Check ECR images
aws ecr describe-images --repository-name churn-prediction-api --region us-east-1
aws ecr describe-images --repository-name churn-prediction-frontend --region us-east-1
```
""")

st.markdown("---")

# Monitoring
st.markdown(
    '<div class="section-header">📈 Monitoring & Alerts</div>', unsafe_allow_html=True
)

st.markdown("""
### **GitHub Actions**

**View Workflow Status**:
1. Go to your repository on GitHub
2. Click **"Actions"** tab
3. See all workflow runs
4. Click on a run to see details

**Workflow Badges**:
Add to your `README.md`:
```markdown
![CI/CD](https://github.com/USERNAME/REPO/workflows/CI%2FCD%20Pipeline/badge.svg)
```

**Notifications**:
- Email on failure
- Slack/Discord webhooks
- GitHub mobile notifications

---

### **AWS Monitoring**

**CloudWatch Logs**:
```bash
# View ECS logs
aws logs tail /ecs/churn-prediction-api --follow

# View specific log stream
aws logs get-log-events \\
  --log-group-name /ecs/churn-prediction-api \\
  --log-stream-name <stream-name>
```

**CloudWatch Metrics**:
- API request count
- Response latency (P50, P95, P99)
- Error rate
- Container CPU/memory

**CloudWatch Alarms**:
```bash
# Create alarm for high error rate
aws cloudwatch put-metric-alarm \\
  --alarm-name api-high-error-rate \\
  --comparison-operator GreaterThanThreshold \\
  --evaluation-periods 2 \\
  --metric-name 5XXError \\
  --namespace AWS/ApiGateway \\
  --period 300 \\
  --statistic Sum \\
  --threshold 10
```

**ECS Service Health**:
```bash
# Check service status
aws ecs describe-services \\
  --cluster churn-prediction-cluster \\
  --services churn-api-service
```
""")

st.markdown("---")

# Troubleshooting
st.markdown(
    '<div class="section-header">🔧 Troubleshooting Guide</div>', unsafe_allow_html=True
)

st.markdown("""
### **Common Issues**

**Issue 1: Tests Fail in CI but Pass Locally**

**Symptoms**:
- Tests pass on your machine
- Fail in GitHub Actions

**Solutions**:
```bash
# 1. Check Python version matches
python --version  # Should be 3.11

# 2. Install exact dependencies
pip install -r requirements.txt

# 3. Run tests in same environment
pytest tests/ -v

# 4. Check for hidden files
ls -la tests/
```

---

**Issue 2: ECR Push Fails - "Access Denied"**

**Symptoms**:
```
Error: AccessDeniedException
```

**Solutions**:
```bash
# 1. Verify IAM permissions
aws iam list-attached-user-policies --user-name github-actions-ecr

# 2. Check secrets in GitHub
#    Settings → Secrets → Verify keys exist

# 3. Test AWS credentials locally
aws ecr get-login-password --region us-east-1

# 4. Ensure ECR repository exists
aws ecr describe-repositories --repository-names churn-prediction-api
```

---

**Issue 3: Docker Build Fails**

**Symptoms**:
```
Error: failed to solve with frontend dockerfile.v0
```

**Solutions**:
```bash
# 1. Test build locally
docker build -t test-image .

# 2. Check .dockerignore
cat .dockerignore

# 3. Ensure all source files committed
git status

# 4. Check Dockerfile syntax
docker build --no-cache -t test-image .
```

---

**Issue 4: Performance Tests Fail**

**Symptoms**:
```
AssertionError: Prediction took 150ms, should be under 100ms
```

**Solutions**:
- GitHub Actions runners can be slower
- Adjust thresholds in `test_performance.py`
- Use `pytest-benchmark` for relative comparisons
- Consider marking as warning instead of failure

---

**Issue 5: Test Model Not Found**

**Symptoms**:
```
FileNotFoundError: tests/test_model.pkl
```

**Solutions**:
```bash
# 1. Create test model
python tests/create_test_model.py

# 2. Commit test model
git add tests/test_model.pkl
git commit -m "Add test model"
git push
```
""")

st.markdown("---")

# Best Practices
st.markdown(
    '<div class="section-header">✨ Best Practices</div>', unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### **Development**
    - ✅ Write tests for new features
    - ✅ Run tests locally before pushing
    - ✅ Keep test model up to date
    - ✅ Use feature branches
    - ✅ Meaningful commit messages
    
    ### **Security**
    - ✅ Rotate AWS keys every 90 days
    - ✅ Use least privilege IAM policies
    - ✅ Never commit secrets to git
    - ✅ Enable branch protection
    - ✅ Require PR reviews
    
    ### **Testing**
    - ✅ Maintain >80% coverage
    - ✅ Test edge cases
    - ✅ Add performance benchmarks
    - ✅ Update tests with code changes
    - ✅ Mock external dependencies
    """)

with col2:
    st.markdown("""
    ### **Deployment**
    - ✅ Deploy to staging first
    - ✅ Monitor after deployment
    - ✅ Keep rollback plan ready
    - ✅ Tag releases semantically
    - ✅ Document breaking changes
    
    ### **Monitoring**
    - ✅ Set up CloudWatch alarms
    - ✅ Track key metrics
    - ✅ Log important events
    - ✅ Regular health checks
    - ✅ Incident response plan
    
    ### **Documentation**
    - ✅ Update docs with changes
    - ✅ Document API changes
    - ✅ Keep README current
    - ✅ Add workflow badges
    - ✅ Maintain changelog
    """)

st.markdown("---")

# Key Files Reference
st.markdown(
    '<div class="section-header">📁 Key Files Reference</div>', unsafe_allow_html=True
)

st.markdown("""
### **Workflow Files**
- `.github/workflows/ci-cd-pipeline.yml` - Main CI/CD workflow with S3 + ECR
- `.github/workflows/test.yml` - Standalone test workflow

### **Test Files**
- `tests/test_data_processing.py` - Data loading & preprocessing tests
- `tests/test_feature_engineering.py` - Feature engineering tests
- `tests/test_inference.py` - Inference engine tests
- `tests/test_edge_cases.py` - Edge case tests
- `tests/test_performance.py` - Performance benchmarks
- `test_api.py` - API endpoint tests
- `tests/create_test_model.py` - Test model generator

### **Deployment Files**
- `Dockerfile` - API container image definition
- `Dockerfile.streamlit` - Frontend container image definition
- `docker-compose.yml` - Local testing with both services
- `.dockerignore` - Build optimization

### **Configuration Files**
- `requirements.txt` - API Python dependencies
- `requirements_streamlit.txt` - Frontend Python dependencies
- `pyproject.toml` - Project metadata and tool configuration
- `.env` - Environment variables (local dev)

### **AWS Resource Scripts**
- `src/utils/s3_handler.py` - S3 operations (upload/download models)
- `src/utils/upload_to_s3.py` - Model upload utility
- `scripts/test_s3_integration.py` - Test S3 connectivity

### **Model Files**
- `models/churn_model_production.pkl` - Local production model
- `tests/test_model.pkl` - Test model for CI/CD
- `S3: s3://churn-project-model/models/churn_model_production.pkl` - Cloud storage

### **Documentation Files**
- `README.md` - Project overview
- `DEPLOYMENT.md` - Deployment instructions
- `API_EXAMPLES.md` - API usage examples
- `streamlit_docs/` - This documentation site

### **Environment Variables** (GitHub Secrets)
- `AWS_ACCESS_KEY_ID` - AWS credentials
- `AWS_SECRET_ACCESS_KEY` - AWS credentials
- `AWS_REGION` - Deployment region (us-east-1)
- `S3_BUCKET_NAME` - Model storage bucket
- `ECR_REPOSITORY` - API image repository
- `ECR_REPOSITORY_STREAMLIT` - Frontend image repository
""")

st.markdown("---")

# Summary
st.markdown("""
## 🎯 Summary

### **What You Get**
- ✅ **140+ automated tests** covering all functionality
- ✅ **Complete CI/CD pipeline** with GitHub Actions
- ✅ **S3 model storage** for centralized ML artifacts
- ✅ **Automated deployment** to Amazon ECR (API + Frontend)
- ✅ **Security scanning** (dependencies + code + Docker)
- ✅ **Performance validation** (<100ms latency)
- ✅ **Zero-downtime deployment** to ECS/Fargate
- ✅ **Full traceability** (commit SHA tags)

### **Pipeline Guarantees**
- 🛡️ **Only tested code reaches production**
- 🚀 **Fast feedback** (~15-18 minutes)
- 🔒 **Security validated** automatically
- 📊 **Performance verified** before deploy
- 🐳 **Containers tested** before push
- ☁️ **Deployment automated** on main branch
- 🗄️ **Model verified** in S3 before deployment

### **AWS Resources Created**
1. **S3 Bucket**: `churn-project-model`
   - Stores trained model: `models/churn_model_production.pkl`
   - Versioned for rollback capability
   - Accessed by API containers at startup

2. **ECR Repository 1**: `churn-prediction-api`
   - FastAPI backend container
   - Loads model from S3
   - Port 8000

3. **ECR Repository 2**: `churn-prediction-frontend`
   - Streamlit frontend container
   - Connects to API backend
   - Port 8501

### **Required Setup**
1. Create S3 bucket (`churn-project-model`)
2. Upload trained model to S3
3. Create ECR repositories (API + Frontend)
4. Create IAM user with S3 + ECR permissions
5. Add GitHub secrets (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
6. Create test model (`python tests/create_test_model.py`)
7. Commit workflow files
8. Push to GitHub

### **Deployment Flow**
```
Code Push → Tests (140+) → Build Docker → Verify S3 Model → 
Push to ECR (API) → Push to ECR (Frontend) → Ready for Production!
```

### **Result**
**Fully automated, production-ready deployment pipeline!** 🎉

Every push to `main` that passes all 140+ tests automatically:
- ✅ Verifies model exists in S3
- ✅ Pushes API container to ECR
- ✅ Pushes Frontend container to ECR
- ✅ Tags images with commit SHA
- ✅ Ready for ECS/Fargate deployment

**Complete Stack Ready**:
- 🗄️ Model: `s3://churn-project-model/models/churn_model_production.pkl`
- 🐳 API: `<account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-api:latest`
- 🎨 Frontend: `<account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-frontend:latest`
""")

st.markdown("---")

# Footer
st.markdown(
    """
<div style="text-align: center; padding: 2rem; background-color: #F5F5F5; border-radius: 8px;">
    <h3>🚀 CI/CD Pipeline Complete</h3>
    <p><strong>From Code to Cloud in 15-18 Minutes</strong></p>
    <p style="color: #666;">Automated Testing → Docker Build → S3 Model Verification → AWS ECR → Production Ready</p>
    <p style="font-size: 0.9rem; margin-top: 1rem;">
        📊 140+ Tests | 🗄️ S3 Storage | 🐳 2 ECR Images | 🔒 Security Scanned | ⚡ <100ms Latency | ☁️ AWS Ready
    </p>
    <p style="font-size: 0.85rem; margin-top: 0.5rem; color: #888;">
        Model: S3 | API: ECR | Frontend: ECR | Deployment: ECS/Fargate/EC2
    </p>
</div>
""",
    unsafe_allow_html=True,
)
