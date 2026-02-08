# 🔮 Churn Prediction Streamlit Frontend

Interactive web interface for the Customer Churn Prediction API.

## Features

### 🎯 Single Prediction
- Enter individual customer details
- Get instant churn prediction
- View churn probability, prediction, and risk level
- Interactive form with validation

### 📊 Batch Prediction
- Upload CSV file with multiple customers
- Process hundreds or thousands of predictions at once
- Download results as CSV
- View summary statistics and risk distribution
- Sample CSV template included

### 🔧 System Monitoring
- Real-time API health status
- Model information display
- Configurable API endpoint

## Quick Start

### Local Development

1. **Install dependencies:**
```bash
pip install -r requirements_streamlit.txt
```

2. **Set API URL (optional):**
```bash
export API_URL=http://localhost:8000
```

3. **Run the app:**
```bash
streamlit run streamlit_app.py
```

4. **Open browser:**
Navigate to http://localhost:8501

### Docker

**Build and run:**
```bash
docker build -f Dockerfile.streamlit -t churn-frontend .
docker run -p 8501:8501 -e API_URL=http://api:8000 churn-frontend
```

### Docker Compose (Full Stack)

**Run both API and frontend:**
```bash
docker-compose up -d
```

Access:
- Frontend: http://localhost:8501
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_URL` | FastAPI backend URL | `http://localhost:8000` |
| `STREAMLIT_SERVER_PORT` | Port to run Streamlit | `8501` |
| `STREAMLIT_SERVER_ADDRESS` | Server address | `0.0.0.0` |

## CSV Format for Batch Prediction

Required columns:
```csv
customerid,age,gender,tenure,usage_frequency,support_calls,payment_delay,subscription_type,contract_length,total_spend,last_interaction
CUST001,35,Male,24,15,3,5,Premium,Annual,1250.50,10
CUST002,42,Female,36,8,5,15,Basic,Monthly,450.00,30
```

### Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| customerid | string | Unique customer identifier |
| age | integer | Customer age (18-100) |
| gender | string | Male or Female |
| tenure | integer | Months as customer (0-120) |
| usage_frequency | integer | Usage frequency (0-100) |
| support_calls | integer | Number of support calls (0-50) |
| payment_delay | integer | Days of payment delay (0-90) |
| subscription_type | string | Basic, Standard, or Premium |
| contract_length | string | Monthly, Quarterly, or Annual |
| total_spend | float | Total amount spent ($) |
| last_interaction | integer | Days since last interaction (0-365) |

## Deployment

### AWS ECS Deployment

1. **Pull image from ECR:**
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker pull <account-id>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-frontend:latest
```

2. **Create ECS Task Definition:**
```json
{
  "family": "churn-frontend",
  "containerDefinitions": [
    {
      "name": "frontend",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-frontend:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "API_URL",
          "value": "http://churn-api:8000"
        }
      ]
    }
  ]
}
```

3. **Create ECS Service:**
```bash
aws ecs create-service \
  --cluster churn-cluster \
  --service-name churn-frontend \
  --task-definition churn-frontend \
  --desired-count 1 \
  --launch-type FARGATE
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn-frontend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: churn-frontend
  template:
    metadata:
      labels:
        app: churn-frontend
    spec:
      containers:
      - name: frontend
        image: <account-id>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-frontend:latest
        ports:
        - containerPort: 8501
        env:
        - name: API_URL
          value: "http://churn-api-service:8000"
---
apiVersion: v1
kind: Service
metadata:
  name: churn-frontend-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8501
  selector:
    app: churn-frontend
```

## Usage Examples

### Single Prediction

1. Navigate to the "Single Prediction" tab
2. Fill in customer details:
   - Customer ID: CUST001
   - Age: 35
   - Gender: Male
   - Tenure: 24 months
   - etc.
3. Click "Predict Churn"
4. View results showing:
   - Churn probability percentage
   - Will Churn / Will Stay prediction
   - Risk level (Low/Medium/High)

### Batch Prediction

1. Navigate to the "Batch Prediction" tab
2. Download the sample CSV template
3. Fill in your customer data
4. Upload the CSV file
5. Click "Predict Batch"
6. View summary statistics and detailed results
7. Download results as CSV

## Troubleshooting

### API Connection Issues

**Error:** "API is Offline"

**Solutions:**
1. Check API URL in sidebar configuration
2. Ensure API container is running: `docker ps`
3. Check API health: `curl http://localhost:8000/health`
4. Review API logs: `docker logs churn-api`

### File Upload Issues

**Error:** "Error reading CSV file"

**Solutions:**
1. Ensure CSV has all required columns
2. Check for proper CSV formatting
3. Verify no special characters in data
4. Download and use the sample template

### Performance Issues

**Slow batch predictions:**
- API may be processing large batch
- Check API container resources
- Consider splitting large files into smaller batches

## Screenshots

### Single Prediction Interface
![Single Prediction](docs/images/single_prediction.png)

### Batch Prediction Interface
![Batch Prediction](docs/images/batch_prediction.png)

### System Status
![System Status](docs/images/system_status.png)

## Development

### Project Structure
```
streamlit_app.py          # Main Streamlit application
Dockerfile.streamlit      # Docker configuration
requirements_streamlit.txt # Python dependencies
```

### Adding New Features

1. **Edit streamlit_app.py**
2. **Test locally:**
```bash
streamlit run streamlit_app.py
```

3. **Build and test Docker image:**
```bash
docker build -f Dockerfile.streamlit -t churn-frontend:dev .
docker run -p 8501:8501 churn-frontend:dev
```

## Support

For issues or questions:
- Check API logs: `docker logs churn-api`
- Check Frontend logs: `docker logs churn-frontend`
- Review API documentation: http://localhost:8000/docs
- Contact system administrator

## License

Part of the Churn Prediction System
