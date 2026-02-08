# GitHub Secrets Configuration for CI/CD Pipeline

## 📋 Overview

This document lists all the GitHub Secrets you need to configure in your repository to enable the complete CI/CD pipeline, including Docker image building and deployment to Amazon ECR.

## 🔐 Required Secrets

### **1. AWS Credentials (REQUIRED for ECR Deployment)**

#### `AWS_ACCESS_KEY_ID`
- **Description**: Your AWS IAM user access key ID
- **Format**: String (e.g., `AKIAIOSFODNN7EXAMPLE`)
- **How to get it**:
  1. Go to AWS IAM Console
  2. Create a new IAM user or use existing one
  3. Attach policy: `AmazonEC2ContainerRegistryPowerUser` + `AmazonECS_FullAccess`
  4. Create access key under "Security credentials"
  5. Copy the Access Key ID

#### `AWS_SECRET_ACCESS_KEY`
- **Description**: Your AWS IAM user secret access key
- **Format**: String (e.g., `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`)
- **How to get it**:
  1. Same IAM user as above
  2. Copy the Secret Access Key when creating the access key
  3. ⚠️ **IMPORTANT**: Save it immediately, you can't retrieve it later!

---

### **2. ECS Deployment (OPTIONAL - Only if auto-deploying to ECS)**

#### `ECS_CLUSTER`
- **Description**: Name of your ECS cluster
- **Format**: String (e.g., `churn-prediction-cluster`)
- **How to get it**:
  1. Go to AWS ECS Console
  2. Copy your cluster name

#### `ECS_SERVICE`
- **Description**: Name of your ECS service
- **Format**: String (e.g., `churn-api-service`)
- **How to get it**:
  1. Go to your ECS cluster
  2. Copy your service name

---

### **3. GitHub Token (OPTIONAL - For Release Creation)**

#### `GITHUB_TOKEN`
- **Description**: GitHub automatically provides this token
- **Format**: Automatically available in workflows
- **Setup**: No manual setup required! GitHub provides this automatically
- **Note**: Used for creating releases when tagging versions

---

## 🛠️ How to Add Secrets to GitHub Repository

### Step 1: Navigate to Repository Settings
```
Your Repository → Settings → Secrets and variables → Actions
```

### Step 2: Add New Repository Secret
1. Click **"New repository secret"**
2. Enter **Name** (e.g., `AWS_ACCESS_KEY_ID`)
3. Enter **Value** (your actual secret value)
4. Click **"Add secret"**

### Step 3: Repeat for All Required Secrets
Add all the secrets listed above following the same process.

---

## 🔑 Required IAM Permissions

Your AWS IAM user needs these permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:GetRepositoryPolicy",
        "ecr:DescribeRepositories",
        "ecr:ListImages",
        "ecr:DescribeImages",
        "ecr:BatchGetImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:PutImage"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecs:UpdateService",
        "ecs:DescribeServices",
        "ecs:DescribeTaskDefinition",
        "ecs:RegisterTaskDefinition"
      ],
      "Resource": "*"
    }
  ]
}
```

### Recommended: Use AWS Managed Policies
Instead of creating custom policy, attach these AWS managed policies:
- `AmazonEC2ContainerRegistryPowerUser`
- `AmazonECS_FullAccess` (only if deploying to ECS)

---

## 📝 Complete Secrets Checklist

### Minimum Required (for ECR push only):
- [x] `AWS_ACCESS_KEY_ID`
- [x] `AWS_SECRET_ACCESS_KEY`

### Optional (for automatic ECS deployment):
- [ ] `ECS_CLUSTER`
- [ ] `ECS_SERVICE`

### Automatically Provided:
- [x] `GITHUB_TOKEN` (no setup needed)

---

## 🧪 Testing Your Setup

### 1. Create a Test IAM User
```bash
# AWS CLI commands
aws iam create-user --user-name github-actions-ecr

# Attach policy
aws iam attach-user-policy \
  --user-name github-actions-ecr \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser

# Create access key
aws iam create-access-key --user-name github-actions-ecr
```

### 2. Test ECR Access
```bash
# Test authentication
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Test list repositories
aws ecr describe-repositories --region us-east-1
```

### 3. Create ECR Repository
```bash
# Create repository for your images
aws ecr create-repository \
  --repository-name churn-prediction-api \
  --region us-east-1
```

---

## 🔄 Workflow Trigger Configuration

The CI/CD pipeline triggers on:

### Push to Main/Develop Branches
```yaml
on:
  push:
    branches: [ main, develop ]
```

### Pull Requests
```yaml
on:
  pull_request:
    branches: [ main, develop ]
```

### Manual Trigger (Optional)
Add this to enable manual workflow runs:
```yaml
on:
  workflow_dispatch:
```

---

## 🚀 Deployment Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Code Push to GitHub                                      │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│ 2. GitHub Actions Workflow Triggered                        │
│    - Code Quality Checks (Black, Flake8)                    │
│    - Unit Tests (Data Processing, Features, Inference)      │
│    - Integration Tests (API Endpoints)                      │
│    - Edge Case Tests (Error Handling)                       │
│    - Performance Tests (Latency, Throughput)                │
│    - Security Scan (Safety, Bandit)                         │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ├── Tests Failed? → Stop (notify team)
                 │
┌────────────────▼────────────────────────────────────────────┐
│ 3. Build Docker Image                                        │
│    - Multi-stage build                                       │
│    - Security scan with Trivy                                │
│    - Test container health                                   │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│ 4. Push to Amazon ECR (if main branch)                      │
│    - AWS Authentication (uses secrets)                       │
│    - Tag with: latest, sha, branch                           │
│    - Push image layers                                       │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│ 5. Deploy to ECS (optional)                                 │
│    - Update ECS service                                      │
│    - Force new deployment                                    │
│    - Health checks                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 What Gets Tested

### ✅ Data Processing Tests
- Load CSV files correctly
- Handle missing values
- Clean and preprocess data
- Validate data quality

### ✅ Feature Engineering Tests
- Create 10 engineered features
- Encode categorical variables
- Scale numerical features
- Maintain data consistency

### ✅ Inference Engine Tests
- Preprocess new data
- Load models from file
- Single customer predictions
- Batch predictions
- Risk level calculations

### ✅ API Endpoint Tests
- Root endpoint (/)
- Health check (/health)
- Readiness (/ready)
- Single prediction (/predict)
- Batch prediction (/predict/batch)
- Model info (/model/info)

### ✅ Edge Case Tests
- Missing values
- Invalid data types
- Out of range values
- Empty inputs
- Malformed data
- Special characters
- Boundary values
- Large batches (1000+ records)
- Duplicate IDs

### ✅ Performance Tests
- Prediction latency (<100ms)
- Batch throughput
- Memory efficiency
- Concurrent requests

### ✅ Security Tests
- Dependency vulnerability scan
- Code security analysis
- Docker image CVE scan

---

## 🔒 Security Best Practices

### 1. Rotate Secrets Regularly
```bash
# Rotate every 90 days
aws iam create-access-key --user-name github-actions-ecr
aws iam delete-access-key --user-name github-actions-ecr --access-key-id OLD_KEY
```

### 2. Use Least Privilege
Only grant permissions absolutely necessary:
- ECR: Push/Pull only
- ECS: Update service only

### 3. Enable MFA for IAM User
```bash
aws iam enable-mfa-device \
  --user-name github-actions-ecr \
  --serial-number arn:aws:iam::ACCOUNT:mfa/DEVICE \
  --authentication-code1 CODE1 \
  --authentication-code2 CODE2
```

### 4. Monitor Access
- Enable CloudTrail
- Set up CloudWatch alarms
- Review access logs regularly

---

## 🆘 Troubleshooting

### Error: "Access Denied" when pushing to ECR
**Solution**: Check IAM permissions, ensure `AmazonEC2ContainerRegistryPowerUser` is attached

### Error: "Repository does not exist"
**Solution**: Create ECR repository first:
```bash
aws ecr create-repository --repository-name churn-prediction-api
```

### Error: "Invalid AWS credentials"
**Solution**: Verify secrets are correctly set in GitHub:
1. Check secret names match exactly
2. Ensure no extra spaces or newlines
3. Re-create access keys if needed

### Error: "Docker image too large"
**Solution**: Check .dockerignore file includes:
- `.git`
- `notebooks/`
- `tests/`
- `__pycache__/`

---

## 📞 Support

### AWS Support
- AWS Console: https://console.aws.amazon.com
- AWS Documentation: https://docs.aws.amazon.com

### GitHub Actions Support
- GitHub Actions Docs: https://docs.github.com/en/actions
- Workflow Syntax: https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions

---

## ✅ Post-Setup Verification

After adding all secrets, verify by:

1. **Push a test commit to a non-main branch**
   - Should run all tests but NOT deploy

2. **Merge to main branch**
   - Should run all tests
   - Build Docker image
   - Push to ECR
   - (Optionally) Deploy to ECS

3. **Check workflow results**
   - Go to "Actions" tab in GitHub
   - Verify all jobs passed
   - Check ECR for new image

4. **Verify ECR image**
```bash
aws ecr list-images --repository-name churn-prediction-api --region us-east-1
```

---

## 🎉 Success Indicators

You'll know everything is working when:
- ✅ All tests pass in GitHub Actions
- ✅ Docker image builds successfully
- ✅ Image appears in Amazon ECR
- ✅ Image has correct tags (latest, sha, branch)
- ✅ (Optional) ECS service updates automatically
- ✅ No error notifications

---

**Last Updated**: February 2026  
**Maintained By**: Your Team  
**Questions?**: Open an issue in the repository
