"""
Page 4: Inference Engine Documentation
Covers the production inference module
"""

import streamlit as st

st.set_page_config(page_title="Inference Engine", page_icon="🔮", layout="wide")

# Custom CSS
st.markdown(
    """
<style>
    .file-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #F57C00;
        margin-top: 2rem;
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #F57C00;
    }
    .function-box {
        background-color: #E3F2FD;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #1E88E5;
    }
    .class-box {
        background-color: #F3E5F5;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #9C27B0;
    }
    .function-name {
        font-family: 'Courier New', monospace;
        font-weight: bold;
        color: #0D47A1;
        font-size: 1.2rem;
    }
    .class-name {
        font-family: 'Courier New', monospace;
        font-weight: bold;
        color: #6A1B9A;
        font-size: 1.3rem;
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
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.title("🔮 Inference Engine")
st.markdown("### Production-ready module for making predictions")
st.markdown("---")

# Overview
st.markdown("""
## 📋 Overview

The inference engine is the **production component** that:
- 🔄 Preprocesses new data (same as training)
- 📦 Loads trained models from MLflow or files
- 🎯 Makes predictions for single or batch customers
- 📊 Calculates risk levels and probabilities

**Location**: `src/inference/inference.py`

**Purpose**: Bridge between trained model and production API

**Key Feature**: Ensures inference preprocessing matches training exactly!
""")

st.markdown("---")

# ============================================================================
# FILE: inference.py
# ============================================================================

st.markdown(
    '<div class="file-header">📄 File: src/inference/inference.py</div>',
    unsafe_allow_html=True,
)

st.markdown("""
**Purpose**: Complete production inference pipeline

**Location**: `src/inference/inference.py`

**Dependencies**: `pandas`, `numpy`, `mlflow`, `joblib`, `sklearn`

**Key Classes**:
1. `InferencePreprocessor` - Preprocessing pipeline
2. `ModelLoader` - Load models from various sources
3. `ChurnPredictor` - Main prediction engine
""")

st.markdown("---")

# Class 1: InferencePreprocessor
st.markdown('<div class="class-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="class-name">📦 Class: InferencePreprocessor</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
**Purpose**: Preprocess raw customer data for prediction

**Why Critical**: Must match training preprocessing EXACTLY or predictions will be wrong!

**Methods**:
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Method 1
st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">🔹 clean_column_names(df)</div>', unsafe_allow_html=True
)
st.markdown(
    """
**What it does**: Standardizes column names (same as training)

**Transformations**:
- "Customer ID" → "customerid"
- Removes special characters (/, spaces)
- Converts to lowercase
- Uses regex for consistency

**Example**:
```python
Input: ["Customer ID", "Total Spend", "Usage/Frequency"]
Output: ["customerid", "total_spend", "usage_frequency"]
```
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Method 2
st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">🔹 encode_features(df)</div>', unsafe_allow_html=True
)
st.markdown(
    """
**What it does**: Encode categorical features (same as training)

**Encoding Steps**:
1. **Label Encode Gender**: Male→0, Female→1
2. **One-Hot Encode** all categoricals:
   - Subscription Type → sub_Basic, sub_Standard, sub_Premium
   - Contract Length → contract_Monthly, contract_Quarterly, contract_Annual
   - Tenure Category → tenuregroup_0-12M, tenuregroup_12-24M, etc.
   - Age Group → agegroup_18-30, agegroup_30-40, etc.
   - Spend Category → spendcategory_Low, spendcategory_Medium, spendcategory_High

3. **Drop Original** categorical columns

**Example**:
```python
Input:
  gender: "Male"
  subscription_type: "Premium"

Output:
  gender: 0
  sub_Basic: 0
  sub_Standard: 0
  sub_Premium: 1
```
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Method 3
st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">🔹 align_features(df, expected_features)</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
**What it does**: Ensures DataFrame has EXACT same features as training

**Critical for Production**:
- Adds missing columns with zeros
- Removes extra columns
- Reorders to match training

**Why Needed**:
If training had 45 features but input has 40, model will crash!

**Example**:
```python
Training features: ['age', 'tenure', 'sub_Basic', 'sub_Premium', ...]  # 45 total
Input features: ['age', 'tenure', 'sub_Basic']  # Missing sub_Premium

After alignment: ['age', 'tenure', 'sub_Basic', 'sub_Premium', ...]
# sub_Premium filled with 0
```
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Method 4 - Main preprocess
st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">🔹 preprocess(data, expected_features)</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
**Main Pipeline** - Runs all preprocessing steps in order

**Parameters**:
- <span class="param">data</span>: dict, list, or DataFrame - Raw customer data
- <span class="param">expected_features</span>: list - Feature names from training

**Returns**: <span class="return">pd.DataFrame</span> - Model-ready features

**Pipeline Steps**:
```
1. Clean column names
2. Handle missing values (calls raw_preprocess)
3. Build engineered features (calls build_feature)
4. Encode categorical features
5. Align with expected features
```

**Use Case**:
```python
preprocessor = InferencePreprocessor()

customer = {
    "age": 35,
    "gender": "Male",
    "tenure": 24,
    ...
}

X = preprocessor.preprocess(customer, expected_features)
# X is now ready for model.predict()
```
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Class 2: ModelLoader
st.markdown('<div class="class-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="class-name">📦 Class: ModelLoader</div>', unsafe_allow_html=True
)
st.markdown(
    """
**Purpose**: Load trained models from different sources

**Why**: Flexible model loading for different deployment scenarios

**Static Methods** (call without instance):
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Method 1
st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">🔹 load_from_mlflow_run(run_id, tracking_uri)</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
**What it does**: Load model from specific MLflow run

**Parameters**:
- <span class="param">run_id</span>: MLflow run ID (e.g., "096c396d6abf4a7da6bae24acb8d99fe")
- <span class="param">tracking_uri</span>: MLflow location (default: "./mlruns")

**Returns**: <span class="return">(model, feature_names)</span>

**Use Case**:
```python
# Load exact model from MLflow
model, features = ModelLoader.load_from_mlflow_run(
    run_id='096c396d6abf4a7da6bae24acb8d99fe',
    tracking_uri='./mlruns'
)
```

**When to use**: You know the exact run ID of the model you want
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Method 2
st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">🔹 load_best_from_experiment(experiment_name, tracking_uri, metric)</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
**What it does**: Automatically find and load best model from experiment

**Parameters**:
- <span class="param">experiment_name</span>: Name of MLflow experiment
- <span class="param">tracking_uri</span>: MLflow location
- <span class="param">metric</span>: Metric to optimize (default: "recall")

**Returns**: <span class="return">(model, feature_names, run_id)</span>

**Use Case**:
```python
# Load best model by recall
model, features, run_id = ModelLoader.load_best_from_experiment(
    experiment_name='Colab_GPU_Training',
    metric='recall'
)
print(f"Loaded model {run_id} with best recall")
```

**When to use**: Always load the best performing model (recommended for production!)
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Method 3
st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">🔹 load_from_file(model_path)</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
**What it does**: Load model from .pkl or .joblib file

**Parameters**:
- <span class="param">model_path</span>: Path to model file

**Returns**: <span class="return">(model, feature_names)</span>

**Use Case**:
```python
# Load from file (no MLflow needed)
model, features = ModelLoader.load_from_file('models/model_best.pkl')
```

**When to use**: Deploy without MLflow server (simpler for production)
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Class 3: ChurnPredictor
st.markdown('<div class="class-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="class-name">📦 Class: ChurnPredictor</div>', unsafe_allow_html=True
)
st.markdown(
    """
**Purpose**: Main inference engine - combines everything

**Attributes**:
- `model`: Trained XGBoost model
- `feature_names`: List of expected features
- `threshold`: Classification threshold (default: 0.5)
- `preprocessor`: InferencePreprocessor instance
- `model_info`: Model metadata

**Key Methods**:
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Method 1
st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">🔹 load_model_from_mlflow_run(run_id, tracking_uri)</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
**What it does**: Initialize predictor with model from MLflow run

**Use Case**:
```python
predictor = ChurnPredictor()
predictor.load_model_from_mlflow_run(
    run_id='096c396d6abf4a7da6bae24acb8d99fe'
)
```
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Method 2
st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">🔹 load_best_model_from_experiment(experiment_name, tracking_uri, metric)</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
**What it does**: Initialize predictor with best model from experiment

**Use Case**:
```python
predictor = ChurnPredictor()
predictor.load_best_model_from_experiment(
    experiment_name='Colab_GPU_Training',
    metric='recall'
)
```
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Method 3 - predict
st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">🔹 predict(customer_data)</div>', unsafe_allow_html=True
)
st.markdown(
    """
**What it does**: Predict churn for a single customer

**Parameters**:
- <span class="param">customer_data</span>: dict - Customer information

**Returns**: <span class="return">dict</span> - Prediction result

**Use Case**:
```python
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
```

**Output**:
```python
{
    "customerid": "CUST001",
    "churn_probability": 0.2345,  # 23.45% chance of churn
    "churn_prediction": 0,         # 0=won't churn, 1=will churn
    "risk_level": "Low"            # Low/Medium/High/Critical
}
```

**Pipeline**:
1. Preprocess customer data
2. Model predicts probability
3. Apply threshold (>0.5 = churn)
4. Calculate risk level
5. Return structured result
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Method 4 - predict_batch
st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">🔹 predict_batch(customers)</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
**What it does**: Predict churn for multiple customers at once

**Parameters**:
- <span class="param">customers</span>: list[dict] - List of customer data

**Returns**: <span class="return">dict</span> - Batch prediction results

**Use Case**:
```python
customers = [
    {"age": 35, "gender": "Male", ...},
    {"age": 28, "gender": "Female", ...},
    {"age": 45, "gender": "Male", ...}
]

results = predictor.predict_batch(customers)
```

**Output**:
```python
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
        },
        ...
    ],
    "total_customers": 3,
    "high_risk_count": 1,       # Count of High/Critical
    "churn_rate": 0.3333        # 1 out of 3 predicted to churn
}
```

**Why Batch**:
- Much faster than looping (vectorized operations)
- Efficient for bulk processing
- Used in API batch endpoint
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Method 5 - _calculate_risk_level
st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">🔹 _calculate_risk_level(probability)</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
**What it does**: Convert probability to risk category

**Risk Levels**:
```python
if probability >= 0.8:
    return "Critical"  # Very high risk
elif probability >= 0.6:
    return "High"      # High risk
elif probability >= 0.4:
    return "Medium"    # Medium risk
else:
    return "Low"       # Low risk
```

**Business Use**:
- **Critical (≥80%)**: Immediate action needed (retention call, special offer)
- **High (60-80%)**: Proactive outreach
- **Medium (40-60%)**: Monitor closely
- **Low (<40%)**: Standard customer care
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Method 6 - get_model_info
st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">🔹 get_model_info()</div>', unsafe_allow_html=True
)
st.markdown(
    """
**What it does**: Return model metadata

**Returns**: <span class="return">dict</span> - Model information

**Output Example**:
```python
{
    "source": "mlflow_experiment",
    "experiment_name": "Colab_GPU_Training",
    "run_id": "096c396d6abf4a7da6bae24acb8d99fe",
    "feature_count": 45,
    "threshold": 0.5,
    "model_type": "XGBClassifier",
    "model_loaded": True
}
```

**Use**: API health checks and debugging
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Convenience Functions
st.markdown("""
## 🎯 Convenience Functions

Quick setup functions for common use cases:
""")

st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">🔹 create_predictor_from_mlflow()</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
**One-line setup** from MLflow:

```python
# Option 1: Specific run
predictor = create_predictor_from_mlflow(
    run_id='096c396d6abf4a7da6bae24acb8d99fe',
    threshold=0.5
)

# Option 2: Best from experiment (recommended)
predictor = create_predictor_from_mlflow(
    experiment_name='Colab_GPU_Training',
    metric='recall',
    threshold=0.5
)

# Now ready to predict!
result = predictor.predict(customer_data)
```
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">🔹 create_predictor_from_file()</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
**One-line setup** from file:

```python
predictor = create_predictor_from_file(
    model_path='models/model_best.pkl',
    threshold=0.5
)

# Ready to use
result = predictor.predict(customer_data)
```
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Complete Example
st.markdown("""
## 🚀 Complete Inference Example

```python
from src.inference.inference import create_predictor_from_mlflow

# 1. Initialize predictor (loads best model)
predictor = create_predictor_from_mlflow(
    experiment_name='Colab_GPU_Training',
    metric='recall'
)

# 2. Single prediction
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
print(f"Churn Risk: {result['risk_level']} ({result['churn_probability']:.2%})")
# Output: Churn Risk: Low (23.45%)

# 3. Batch prediction
customers = [customer, another_customer, ...]
batch_results = predictor.predict_batch(customers)
print(f"Total customers: {batch_results['total_customers']}")
print(f"High risk: {batch_results['high_risk_count']}")

# 4. Get model info
info = predictor.get_model_info()
print(f"Model type: {info['model_type']}")
print(f"Features: {info['feature_count']}")
```

## 🎯 Key Takeaways

### **InferencePreprocessor**
- Preprocesses data exactly like training
- Critical for prediction accuracy
- Handles all feature engineering

### **ModelLoader**
- Flexible model loading
- MLflow or file-based
- Automatic best model selection

### **ChurnPredictor**
- Complete prediction engine
- Single and batch predictions
- Risk level calculation
- Production-ready

### **Why This Design?**
1. **Separation of Concerns**: Preprocessing, loading, prediction separate
2. **Reusable**: Each class can be used independently
3. **Testable**: Easy to unit test each component
4. **Flexible**: Multiple model loading options
5. **Production-Ready**: Handles edge cases, errors

**Next Section**: 🌐 API Deployment (using this inference engine in FastAPI!)
""")

st.markdown("---")
st.markdown("👉 **Navigate to 'API Deployment' in the sidebar to continue**")
