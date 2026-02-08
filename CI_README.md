# GitHub Actions CI/CD Workflows

This directory contains GitHub Actions workflows for automated testing, validation, and deployment to Amazon ECR.

---

## 📋 Overview

This project uses multiple GitHub Actions workflows:

1. **`ci-cd-pipeline.yml`** ⭐ - Production CI/CD with comprehensive testing and ECR deployment
2. **`modular-pipeline-test.yml`** - ML pipeline stage-by-stage testing (legacy)
3. **`workflow.yml`** - Simple pytest testing (legacy)

---

## 🚀 Main Workflow: CI/CD Pipeline

### **File**: `workflows/ci-cd-pipeline.yml`

**Purpose**: Complete CI/CD pipeline with 140+ tests and automated deployment to AWS ECR

**Trigger Events**:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

**Duration**: ~15 minutes

### **Pipeline Stages**:

```
1️⃣ Code Quality ✅
   ├── Black formatter check
   ├── Isort import checker
   └── Flake8 linter

2️⃣ Unit Tests (Parallel) ✅
   ├── Data Processing Tests (25+)
   ├── Feature Engineering Tests (30+)
   └── Inference Engine Tests (20+)

3️⃣ Integration Tests ✅
   ├── API Endpoint Tests (6+)
   └── Edge Cases Tests (50+)

4️⃣ Performance & Security ✅
   ├── Performance Tests (10+)
   └── Security Scanning (Bandit, Safety)

5️⃣ Docker Build & Test ✅
   ├── Build multi-stage image
   ├── Test container
   ├── Security scan (Trivy)
   └── Health checks

6️⃣ Deploy to ECR ✅ (main branch only)
   ├── AWS authentication
   ├── Push to ECR
   ├── Tag with commit SHA
   └── Update ECS (optional)

7️⃣ Create Release (tagged versions)
   └── GitHub Release
```

---

## 📊 Test Coverage (140+ Tests)

### **Data Processing Tests** (`tests/test_data_processing.py`)
**Tests**: 25+

- ✅ Data loading from CSV
- ✅ Missing value handling
- ✅ Column name cleaning
- ✅ Data type validation
- ✅ Outlier detection
- ✅ Data quality checks with Great Expectations

### **Feature Engineering Tests** (`tests/test_feature_engineering.py`)
**Tests**: 30+

- ✅ CLV (Customer Lifetime Value) calculation
- ✅ Support efficiency metric
- ✅ Payment reliability score
- ✅ Engagement score
- ✅ Categorical encoding (OneHot, Label)
- ✅ Feature alignment
- ✅ Pipeline reproducibility

### **Inference Engine Tests** (`tests/test_inference.py`)
**Tests**: 20+

- ✅ Preprocessing pipeline
- ✅ Model loading (file & MLflow)
- ✅ Single prediction
- ✅ Batch prediction
- ✅ Risk level calculation (Low/Medium/High/Critical)
- ✅ Model info retrieval

### **API Tests** (`test_api.py`)
**Tests**: 6+

- ✅ Root endpoint (/)
- ✅ Health check (/health)
- ✅ Readiness check (/ready)
- ✅ Single prediction (/predict)
- ✅ Batch prediction (/predict/batch)
- ✅ Model info (/model/info)

### **Edge Cases Tests** (`tests/test_edge_cases.py`)
**Tests**: 50+

- ✅ Missing values (fields, None, empty strings)
- ✅ Invalid data types (strings for numbers)
- ✅ Out of range values (age=200, negative tenure)
- ✅ Empty inputs (empty dict, list)
- ✅ Malformed data (invalid gender, unknown types)
- ✅ Special characters (@#$%)
- ✅ Boundary values (min age, max values, all zeros)
- ✅ Large batches (1000+ records)
- ✅ Duplicate customer IDs

### **Performance Tests** (`tests/test_performance.py`)
**Tests**: 10+

- ✅ Prediction latency (<100ms target)
- ✅ Batch throughput (>10 pred/sec)
- ✅ Memory efficiency (5000 records)
- ✅ Stress testing (1000+ sequential predictions)
- ✅ Concurrency testing

---

## 🔐 Required GitHub Secrets

See [`GITHUB_SECRETS_GUIDE.md`](../GITHUB_SECRETS_GUIDE.md) for detailed setup instructions.

### **Minimum Required** (2 secrets):
```
AWS_ACCESS_KEY_ID     - AWS access key for ECR authentication
AWS_SECRET_ACCESS_KEY - AWS secret key for ECR authentication
```

### **Optional** (for ECS deployment):
```
ECS_CLUSTER  - ECS cluster name for automatic deployment
ECS_SERVICE  - ECS service name for automatic deployment
```

---

## 🛠️ Setup Instructions

### **1. Prerequisites**
- GitHub repository
- AWS account with ECR repository
- IAM user with ECR permissions

### **2. Create ECR Repository**
```bash
aws ecr create-repository \
  --repository-name churn-prediction-api \
  --region us-east-1
```

### **3. Create IAM User**
```bash
# Create user
aws iam create-user --user-name github-actions-ecr

# Attach ECR permissions
aws iam attach-user-policy \
  --user-name github-actions-ecr \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser

# Create access keys
aws iam create-access-key --user-name github-actions-ecr
# Save the AccessKeyId and SecretAccessKey!
```

### **4. Configure GitHub Secrets**

Go to: `Repository → Settings → Secrets and variables → Actions → New repository secret`

Add the following secrets:
- Name: `AWS_ACCESS_KEY_ID`, Value: (your access key)
- Name: `AWS_SECRET_ACCESS_KEY`, Value: (your secret key)

### **5. Create Test Model**
```bash
# Required for CI/CD tests to pass
python tests/create_test_model.py
```

### **6. Update Workflow Configuration** (if needed)

Edit `.github/workflows/ci-cd-pipeline.yml`:

```yaml
env:
  PYTHON_VERSION: '3.11'               # Your Python version
  AWS_REGION: us-east-1                # Your AWS region
  ECR_REPOSITORY: churn-prediction-api # Your ECR repository name
```

### **7. Push to GitHub**
```bash
git add .
git commit -m "Add CI/CD pipeline"
git push origin main
```

The pipeline will automatically trigger and run all 140+ tests before deploying!

---

## 📈 Monitoring & Usage

### **View Workflow Status**

1. Go to your repository on GitHub
2. Click **"Actions"** tab
3. Select a workflow run
4. View job results and logs
5. Download artifacts if needed

### **Workflow Badges**

Add to your `README.md`:
```markdown
![CI/CD Pipeline](https://github.com/USERNAME/REPO/workflows/CI%2FCD%20Pipeline%20-%20Test%20and%20Deploy%20to%20ECR/badge.svg)
```

### **Check Deployment Status**

After pushing to `main` branch:
```bash
# Check ECR images
aws ecr list-images --repository-name churn-prediction-api

# Describe latest image
aws ecr describe-images \
  --repository-name churn-prediction-api \
  --image-ids imageTag=latest
```

---

## 📦 Artifacts & Outputs

### **Test Coverage Reports**
- Uploaded to Codecov (if configured)
- Available as workflow artifacts
- Retention: 7 days

### **Security Reports**
- Bandit report (code security)
- Safety report (dependency vulnerabilities)
- Trivy report (Docker CVEs)
- Available as SARIF files

### **Docker Images**
Pushed to ECR with multiple tags:
- `latest` - Latest main branch build
- `main-<sha>` - Main branch with commit SHA
- `develop-<sha>` - Develop branch with commit SHA
- `v1.2.3` - Semantic version (if tagged)

---

## 🔧 Customization

### **Skip Tests on Certain Files**
```yaml
on:
  push:
    branches: [ main, develop ]
    paths-ignore:
      - '**.md'
      - 'docs/**'
      - 'streamlit_docs/**'
```

### **Add Manual Trigger**
```yaml
on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deployment environment'
        required: true
        default: 'staging'
```

### **Change Python Version**
```yaml
env:
  PYTHON_VERSION: '3.12'  # Update to your version
```

### **Adjust Test Timeouts**
```yaml
- name: Run API tests
  run: pytest test_api.py -v --timeout=60
  timeout-minutes: 5
```

### **Run Tests in Parallel**
```yaml
strategy:
  matrix:
    python-version: ['3.10', '3.11', '3.12']
```

---

## 🚨 Troubleshooting

### **Tests Failing in CI but Passing Locally**

```bash
# Ensure same Python version
python --version

# Install exact dependencies
pip install -r requirements.txt

# Run tests locally
pytest tests/ -v

# Check for hidden files
ls -la tests/
```

### **ECR Push Fails - "Access Denied"**

**Check**:
1. AWS credentials in GitHub Secrets
2. IAM user has ECR permissions
3. ECR repository exists

```bash
# Test AWS credentials
aws ecr get-login-password --region us-east-1

# Verify IAM permissions
aws iam list-attached-user-policies --user-name github-actions-ecr

# Check ECR repository
aws ecr describe-repositories --repository-names churn-prediction-api
```

### **Docker Build Fails**

```bash
# Test build locally
docker build -t test-image .

# Check .dockerignore
cat .dockerignore

# Verify all files committed
git status
```

### **Performance Tests Fail**

GitHub Actions runners can be slower than local machines.

**Solutions**:
- Increase timeout in `test_performance.py`
- Adjust latency threshold (e.g., 150ms instead of 100ms)
- Use relative benchmarks instead of absolute thresholds

### **Test Model Not Found**

```bash
# Create test model
python tests/create_test_model.py

# Commit to repository
git add tests/test_model.pkl
git commit -m "Add test model"
git push
```

### **Workflow Not Triggering**

**Check**:
- Branch protection rules enabled
- Workflow file syntax (use YAML validator)
- Workflows enabled in repository settings
- Push is to correct branch (main/develop)

---

## 📝 Best Practices

### **Development Workflow**
1. Create feature branch
2. Write code + tests
3. Run tests locally
4. Push to GitHub
5. Create Pull Request
6. Review test results
7. Merge to main after approval

### **Branch Strategy**
- `main` → Production deployments (ECR + ECS)
- `develop` → Staging/testing (ECR only)
- `feature/*` → Run tests only (no deployment)

### **Test Coverage Goals**
- Overall coverage: >80%
- Critical modules: >90%
- Add tests for all new features
- Test edge cases and error handling

### **Security Best Practices**
- Rotate AWS keys every 90 days
- Use least privilege IAM policies
- Enable branch protection on main
- Require PR reviews before merge
- Never commit secrets to git

### **Performance Optimization**
- Cache pip dependencies
- Use Docker layer caching
- Run independent jobs in parallel
- Skip unnecessary steps on draft PRs

---

## 📊 Success Criteria

### **Pipeline Passes When**:
- ✅ All 140+ tests pass (100%)
- ✅ Code coverage >70%
- ✅ No critical security vulnerabilities
- ✅ Docker image builds successfully
- ✅ All health checks pass
- ✅ API response time <100ms
- ✅ Image size <2GB

### **Deployment Succeeds When**:
- ✅ Image pushed to ECR
- ✅ Tagged with commit SHA
- ✅ ECS service updated (if configured)
- ✅ Zero downtime deployment

---

## 📊 Metrics Tracked

The pipeline automatically tracks:
- ⏱️ Build time (~15 minutes)
- ✅ Test pass rate (must be 100%)
- 📦 Docker image size
- 🔒 Security vulnerabilities
- 📈 Code coverage
- ⚡ API latency
- 🎯 Prediction accuracy

---

## 🔄 Legacy Workflows

### **2. `modular-pipeline-test.yml` - ML Pipeline Testing**

**Status**: Legacy (optional)  
**Purpose**: Test entire ML pipeline stage-by-stage  
**Duration**: ~10-15 minutes  
**Data**: Uses `dummy_data.csv` (split 70/30)

**Pipeline Stages**:
1. Prepare Data
2. Load Data
3. Validate Data
4. Preprocess
5. Build Features
6. Encode Features
7. Optuna (5 trials)
8. Train Model (2 runs)
9. Save Model
10. Summary

**Note**: This workflow tests the training pipeline. The main CI/CD pipeline tests the inference API.

### **3. `workflow.yml` - Simple Testing**

**Status**: Legacy (optional)  
**Purpose**: Run basic pytest tests  
**Duration**: ~2-3 minutes

---

## 🌟 What Makes This CI/CD Special

### **Comprehensive Testing**
- 140+ automated tests
- Complete coverage (data, features, inference, API, edge cases)
- Performance validation
- Security scanning

### **Production Ready**
- Only deploys if ALL tests pass
- Zero downtime deployment
- Automated rollback capability
- Full traceability (commit SHA tags)

### **Developer Friendly**
- Fast feedback (~15 minutes)
- Clear error messages
- Detailed logs
- Artifact downloads

### **Security First**
- 3-layer security scanning (code, dependencies, Docker)
- No secrets in code
- Least privilege access
- Automated vulnerability detection

---

## 📞 Support & Resources

### **Documentation**
- [`GITHUB_SECRETS_GUIDE.md`](../GITHUB_SECRETS_GUIDE.md) - AWS secrets setup
- [`CI_CD_SETUP_SUMMARY.md`](../CI_CD_SETUP_SUMMARY.md) - Quick start guide
- [`DEPLOYMENT.md`](../DEPLOYMENT.md) - Deployment instructions
- [`streamlit_docs/pages/7_🚀_CI_CD_Deployment.py`](../streamlit_docs/pages/7_🚀_CI_CD_Deployment.py) - Interactive docs

### **Getting Help**
- **Issues**: Open a GitHub issue
- **Questions**: Check existing issues first
- **Updates**: Watch the repository for changes
- **Logs**: Check Actions tab for detailed execution logs

---

## 🎯 Quick Commands Reference

```bash
# Run all tests locally
pytest tests/ -v

# Run specific test suite
pytest tests/test_inference.py -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term

# Create test model
python tests/create_test_model.py

# Build Docker locally
docker build -t churn-api:test .

# Test Docker container
docker run -d -p 8000:8000 churn-api:test
curl http://localhost:8000/health

# Check ECR images
aws ecr list-images --repository-name churn-prediction-api

# Update ECS service
aws ecs update-service \
  --cluster your-cluster \
  --service your-service \
  --force-new-deployment
```

---

**Version**: 2.0.0  
**Last Updated**: February 2026  
**Maintained By**: ML Engineering Team  
**Pipeline Status**: ✅ Production Ready  
**Total Tests**: 140+  
**Deployment**: Fully Automated
