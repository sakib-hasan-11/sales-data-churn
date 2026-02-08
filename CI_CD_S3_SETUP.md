# GitHub Actions CI/CD Pipeline - S3 Configuration

## ✅ Pipeline Updated for S3 Model Loading

Your CI/CD pipeline now supports loading models from S3 in production while using test models during CI tests.

---

## **What Was Changed:**

### 1. **Added S3 Environment Variables**
```yaml
env:
  S3_BUCKET_NAME: churn-project-model
  S3_MODEL_NAME: churn_model_production.pkl
```

### 2. **New Job: Verify S3 Model**
- Checks that your model exists in S3 before deployment
- Validates bucket access
- Displays model metadata (size, last modified)

### 3. **Updated Docker Deployment**
- Passes S3 configuration to Docker build
- ECS deployment will use S3 model automatically
- No model files in Docker image (smaller, faster)

### 4. **Enhanced Notifications**
- Shows S3 bucket and model name in deployment notifications
- Release notes include S3 configuration

---

## **Required GitHub Secrets:**

Configure these in your GitHub repository:
**Settings → Secrets and variables → Actions → New repository secret**

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `AWS_ACCESS_KEY_ID` | Your AWS key | For S3 and ECR access |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret | For S3 and ECR access |
| `ECS_CLUSTER` | Your ECS cluster name | (Optional) For auto-deployment |
| `ECS_SERVICE` | Your ECS service name | (Optional) For auto-deployment |

---

## **Pipeline Flow:**

```
1. Code Quality Checks
   ↓
2. Run All Tests (uses test models)
   ↓
3. Build Docker Image (uses test models for validation)
   ↓
4. Verify S3 Model Exists ← NEW!
   ↓
5. Deploy to ECR (configured for S3)
   ↓
6. Update ECS Service (uses S3 model in production)
```

---

## **How It Works:**

### **During CI Tests:**
- Uses `MODEL_SOURCE=mlflow` or local test models
- Creates test model artifacts: `tests/create_test_model.py`
- Fast, no S3 dependency needed for tests

### **During Production Deployment:**
- Verifies model exists in S3: `s3://churn-project-model/models/churn_model_production.pkl`
- Builds Docker image with S3 configuration
- Deploys to ECR with environment variables:
  ```bash
  MODEL_SOURCE=s3
  S3_BUCKET_NAME=churn-project-model
  S3_MODEL_NAME=churn_model_production.pkl
  ```

### **In ECS Production:**
- Container starts and loads model from S3
- Uses IAM role (recommended) or AWS credentials
- Model cached in memory after first load

---

## **Testing the Pipeline:**

### **1. Test Locally First:**
```bash
# Make sure your model is uploaded
python upload_to_s3.py

# Verify S3 access
aws s3 ls s3://churn-project-model/models/
```

### **2. Test Docker Build:**
```bash
docker build -t churn-api:test .
docker run -p 8000:8000 --env-file .env churn-api:test
curl http://localhost:8000/health
```

### **3. Push to GitHub:**
```bash
git add .
git commit -m "Add S3 model loading to CI/CD pipeline"
git push origin main
```

### **4. Monitor Pipeline:**
- Go to GitHub Actions tab
- Watch the pipeline run
- Check "Verify S3 Model" step succeeds
- Verify deployment completes

---

## **Updating the Model:**

1. **Train new model** (your existing process)
2. **Upload to S3:**
   ```bash
   python upload_to_s3.py
   ```
3. **Deploy automatically:**
   - Push to main branch OR
   - Manually trigger ECS deployment:
     ```bash
     aws ecs update-service --cluster YOUR_CLUSTER --service YOUR_SERVICE --force-new-deployment
     ```

No Docker rebuild needed! 🎉

---

## **Troubleshooting:**

### **Pipeline Fails at "Verify S3 Model":**
```bash
# Check if model exists
aws s3 ls s3://churn-project-model/models/churn_model_production.pkl

# If not found, upload it
python upload_to_s3.py
```

### **AWS Credentials Error:**
- Verify GitHub secrets are set correctly
- Make sure secrets have no extra spaces
- Check IAM permissions include S3 read access

### **ECS Deployment Not Working:**
- Add `ECS_CLUSTER` and `ECS_SERVICE` secrets
- Or remove the ECS update step (it's optional)

---

## **Security Best Practices:**

✅ **Never commit AWS credentials** to code  
✅ **Use GitHub Secrets** for sensitive data  
✅ **Use IAM roles** in ECS (preferred over access keys)  
✅ **Enable S3 versioning** for model rollback  
✅ **Monitor CloudWatch logs** for issues  

---

## **Next Steps:**

1. ✅ Configure GitHub Secrets
2. ✅ Push code to trigger pipeline
3. ✅ Monitor pipeline execution
4. ✅ Verify deployment to ECR
5. ✅ Test production API endpoint

Your pipeline is ready! 🚀
