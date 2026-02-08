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
- ☁️ Deploys to **Amazon ECR** automatically
- 📊 Monitors **performance metrics**

**Location**: `.github/workflows/ci-cd-pipeline.yml`

**Triggers**: Push to main/develop, Pull Requests

**Result**: Only deploys if ALL tests pass! ✅
""")

st.markdown("---")

# Pipeline Overview
st.markdown(
    '<div class="section-header">🔄 Complete Pipeline Flow</div>',
    unsafe_allow_html=True,
)

st.markdown("""
### **11 Sequential Jobs** - All must pass before deployment!

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
🔟 Push to Amazon ECR (main branch only)
    ↓
1️⃣1️⃣ Update ECS Service (optional)
    ↓
✅ Deployed to Production!
```

**Total Pipeline Time**: ~8-12 minutes
**Success Rate Target**: 100% tests pass
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
    '<div class="section-header">☁️ Deployment to Amazon ECR</div>',
    unsafe_allow_html=True,
)

st.markdown("""
### **Automatic Deployment** (Main Branch Only)

When you push to the `main` branch and all tests pass:
""")

st.markdown('<div class="job-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="job-name">Job 10: Deploy to Amazon ECR</div>', unsafe_allow_html=True
)
st.markdown(
    """
**Trigger**: Push to `main` branch + all tests passed

**Steps**:

1. **Authenticate to AWS**
   ```bash
   # Uses GitHub secrets
   aws ecr get-login-password --region us-east-1 | \\
     docker login --username AWS --password-stdin \\
     <account>.dkr.ecr.us-east-1.amazonaws.com
   ```

2. **Tag Docker Image**
   ```bash
   # Multiple tags for traceability
   latest              # Always points to latest main
   main-<commit-sha>   # Specific commit
   v1.2.3             # Semantic version (if tagged)
   ```

3. **Push to ECR**
   ```bash
   docker push <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-api:latest
   docker push <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-api:main-abc123
   ```

4. **Verify Upload**
   ```bash
   # Check image exists in ECR
   aws ecr describe-images \\
     --repository-name churn-prediction-api \\
     --image-ids imageTag=latest
   ```

5. **Update ECS (Optional)**
   ```bash
   # If ECS_CLUSTER and ECS_SERVICE are configured
   aws ecs update-service \\
     --cluster churn-prediction-cluster \\
     --service churn-api-service \\
     --force-new-deployment
   ```

**Result**: 
- ✅ Image available in ECR
- ✅ Tagged with commit SHA for traceability
- ✅ ECS service updated (if configured)
- ✅ Zero downtime deployment
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
9. Build Docker ✅ (3 minutes)
10. Push to ECR ✅ (2 minutes)
11. Update ECS ✅ (1 minute)

Total: ~15 minutes
```

**Result**:
- 🎉 New version deployed to production!
- 📦 Available in ECR as `latest`
- 🚀 ECS automatically pulls new image
- 🔍 Fully tested and validated

---

**Step 4: Verify Deployment**
```bash
# Check ECR
aws ecr list-images --repository-name churn-prediction-api

# Check ECS service
aws ecs describe-services \\
  --cluster churn-prediction-cluster \\
  --services churn-api-service

# Test API
curl https://your-api-url.com/health
curl -X POST https://your-api-url.com/predict -d '{...}'
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

**1. Create ECR Repository**
```bash
aws ecr create-repository \\
  --repository-name churn-prediction-api \\
  --region us-east-1
```

**2. Create IAM User**
```bash
# Create user
aws iam create-user --user-name github-actions-ecr

# Attach policy
aws iam attach-user-policy \\
  --user-name github-actions-ecr \\
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser

# Create access key
aws iam create-access-key --user-name github-actions-ecr
```

**3. Add GitHub Secrets**
- Go to: `Repository → Settings → Secrets and variables → Actions`
- Add `AWS_ACCESS_KEY_ID`
- Add `AWS_SECRET_ACCESS_KEY`

**4. Create Test Model**
```bash
# Required for CI/CD tests
python tests/create_test_model.py
```

**5. Commit All Files**
```bash
git add .github/workflows/ci-cd-pipeline.yml
git add tests/
git commit -m "Add CI/CD pipeline"
git push origin main
```

**6. Monitor First Run**
- Go to: `GitHub → Actions tab`
- Watch pipeline execute
- Verify all jobs pass
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
- `.github/workflows/ci-cd-pipeline.yml` - Main CI/CD workflow
- `.github/workflows/README.md` - Workflow documentation

### **Test Files**
- `tests/test_data_processing.py` - Data loading & preprocessing tests
- `tests/test_feature_engineering.py` - Feature engineering tests
- `tests/test_inference.py` - Inference engine tests
- `tests/test_edge_cases.py` - Edge case tests
- `tests/test_performance.py` - Performance benchmarks
- `test_api.py` - API endpoint tests
- `tests/create_test_model.py` - Test model generator

### **Deployment Files**
- `Dockerfile` - Container image definition
- `docker-compose.yml` - Local testing
- `.dockerignore` - Build optimization
- `ecs-task-definition.json` - ECS configuration

### **Documentation Files**
- `GITHUB_SECRETS_GUIDE.md` - Secrets setup guide
- `CI_CD_SETUP_SUMMARY.md` - Quick start guide
- `DEPLOYMENT.md` - Deployment instructions
- `API_EXAMPLES.md` - API usage examples
""")

st.markdown("---")

# Summary
st.markdown("""
## 🎯 Summary

### **What You Get**
- ✅ **140+ automated tests** covering all functionality
- ✅ **Complete CI/CD pipeline** with GitHub Actions
- ✅ **Automated deployment** to Amazon ECR
- ✅ **Security scanning** (dependencies + code + Docker)
- ✅ **Performance validation** (<100ms latency)
- ✅ **Zero-downtime deployment** to ECS
- ✅ **Full traceability** (commit SHA tags)

### **Pipeline Guarantees**
- 🛡️ **Only tested code reaches production**
- 🚀 **Fast feedback** (~15 minutes)
- 🔒 **Security validated** automatically
- 📊 **Performance verified** before deploy
- 🐳 **Container tested** before push
- ☁️ **Deployment automated** on main branch

### **Required Setup**
1. Create AWS ECR repository
2. Create IAM user with ECR permissions
3. Add GitHub secrets (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
4. Create test model (`python tests/create_test_model.py`)
5. Commit workflow files
6. Push to GitHub

### **Result**
**Fully automated, production-ready deployment pipeline!** 🎉

Every push to `main` that passes all 140+ tests automatically deploys to AWS ECR, ready for production use!
""")

st.markdown("---")

# Footer
st.markdown(
    """
<div style="text-align: center; padding: 2rem; background-color: #F5F5F5; border-radius: 8px;">
    <h3>🚀 CI/CD Pipeline Complete</h3>
    <p><strong>From Code to Cloud in 15 Minutes</strong></p>
    <p style="color: #666;">Automated Testing → Docker Build → AWS ECR → Production</p>
    <p style="font-size: 0.9rem; margin-top: 1rem;">
        📊 140+ Tests | 🔒 Security Scanned | ⚡ <100ms Latency | ☁️ AWS Ready
    </p>
</div>
""",
    unsafe_allow_html=True,
)
