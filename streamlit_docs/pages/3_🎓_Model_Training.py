"""
Page 3: Model Training Documentation
Covers training, hyperparameter tuning, and evaluation
"""

import streamlit as st

st.set_page_config(page_title="Model Training", page_icon="🎓", layout="wide")

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
    .function-name {
        font-family: 'Courier New', monospace;
        font-weight: bold;
        color: #0D47A1;
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
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.title("🎓 Model Training & Evaluation")
st.markdown("### Train, tune, and evaluate XGBoost models with MLflow tracking")
st.markdown("---")

# Overview
st.markdown("""
## 📋 Overview

This section covers the complete model training pipeline:
- 🎯 **Hyperparameter Tuning**: Optuna for automatic optimization
- 🚂 **Model Training**: XGBoost with MLflow tracking
- 📊 **Evaluation**: Comprehensive metrics and model saving

**Location**: `src/training/`

**Tools Used**: XGBoost, Optuna,MLflow, Scikit-learn
""")

st.markdown("---")

# ============================================================================
# FILE 1: optuna_tuning.py
# ============================================================================

st.markdown(
    '<div class="file-header">📄 File: src/training/optuna_tuning.py</div>',
    unsafe_allow_html=True,
)

st.markdown("""
**Purpose**: Find optimal XGBoost hyperparameters using Optuna

**Location**: `src/training/optuna_tuning.py`

**Dependencies**: `optuna`, `xgboost`, `sklearn`

**Why Optuna**: Automatically finds best hyperparameters (learning rate, depth, etc.)
""")

# Function 1
st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown('<div class="function-name">🔹 objective()</div>', unsafe_allow_html=True)
st.markdown(
    """
**Description**: Optuna objective function that defines hyperparameter search space

**Parameters**:
- <span class="param">trial</span>: `optuna.Trial` - Optuna trial object for suggesting parameters

**Returns**: <span class="return">float</span> - Metric to optimize (e.g., recall score)

**What it does**:
1. **Suggests Hyperparameters** to try:
```python
params = {
    'max_depth': trial.suggest_int('max_depth', 3, 10),
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
    'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
    'gamma': trial.suggest_float('gamma', 0, 0.5),
    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0)
}
```

**Hyperparameter Explanations**:
- **max_depth** (3-10): Tree depth - deeper = more complex
- **learning_rate** (0.01-0.3): Step size - smaller = slower but more stable
- **n_estimators** (100-1000): Number of trees - more = better but slower
- **min_child_weight** (1-7): Minimum data in leaf - controls overfitting
- **gamma** (0-0.5): Minimum loss reduction - higher = more conservative
- **subsample** (0.5-1.0): Fraction of data per tree - prevents overfitting
- **colsample_bytree** (0.5-1.0): Fraction of features per tree
- **reg_alpha** (0-1): L1 regularization
- **reg_lambda** (0-1): L2 regularization

2. **Trains Model** with suggested parameters
3. **Evaluates** on validation set
4. **Returns Score** (Optuna maximizes or minimizes this)

**Use Case**:
```python
# Optuna automatically calls this function many times
# with different hyperparameter combinations
study = optuna.create_study(direction='maximize')  # maximize recall
study.optimize(objective, n_trials=100)

print(f"Best recall: {study.best_value}")
print(f"Best params: {study.best_params}")
```
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Function 2
st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">🔹 tune_hyperparameters()</div>', unsafe_allow_html=True
)
st.markdown(
    """
**Description**: Main function to run Optuna hyperparameter optimization

**Parameters**:
- <span class="param">X_train</span>: Training features
- <span class="param">y_train</span>: Training labels
- <span class="param">X_val</span>: Validation features
- <span class="param">y_val</span>: Validation labels
- <span class="param">n_trials</span>: Number of optimization trials (default: 100)
- <span class="param">metric</span>: Metric to optimize ('recall', 'precision', 'f1', etc.)

**Returns**: <span class="return">dict</span> - Best hyperparameters found

**Use Case**:
```python
best_params = tune_hyperparameters(
    X_train, y_train,
    X_val, y_val,
    n_trials=100,
    metric='recall'  # Optimize for recall (find churners)
)
```

**What happens**:
1. Creates Optuna study
2. Runs 100 trials (tries 100 different hyperparameter combinations)
3. Each trial:
   - Suggests new parameters
   - Trains XGBoost model
   - Evaluates on validation set
   - Records score
4. Returns best parameters found
5. Typical runtime: 10-30 minutes for 100 trials

**Output Example**:
```
Trial 1: recall=0.78 (params: max_depth=5, lr=0.1...)
Trial 2: recall=0.82 (params: max_depth=7, lr=0.05...)
...
Trial 100: recall=0.79

Best trial: #23 with recall=0.87
Best params: {
    'max_depth': 6,
    'learning_rate': 0.08,
    'n_estimators': 500,
    ...
}
```
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# FILE 2: mlflow_training.py
# ============================================================================

st.markdown(
    '<div class="file-header">📄 File: src/training/mlflow_training.py</div>',
    unsafe_allow_html=True,
)

st.markdown("""
**Purpose**: Train XGBoost model with best parameters and track with MLflow

**Location**: `src/training/mlflow_training.py`

**Dependencies**: `xgboost`, `mlflow`, `sklearn`

**Why MLflow**: Tracks experiments, parameters, metrics, and saves models
""")

# Function 1
st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">🔹 train_with_mlflow()</div>', unsafe_allow_html=True
)
st.markdown(
    """
**Description**: Train XGBoost model and log everything to MLflow

**Parameters**:
- <span class="param">X_train</span>: Training features
- <span class="param">y_train</span>: Training labels
- <span class="param">X_test</span>: Test features
- <span class="param">y_test</span>: Test labels
- <span class="param">params</span>: `dict` - XGBoost hyperparameters (from Optuna)
- <span class="param">threshold</span>: `float` - Classification threshold (default: 0.5)
- <span class="param">experiment_name</span>: `str` - MLflow experiment name

**Returns**: <span class="return">tuple</span> - (trained_model, run_id)

**Use Case**:
```python
# Use best params from Optuna
model, run_id = train_with_mlflow(
    X_train, y_train,
    X_test, y_test,
    params=best_params,
    threshold=0.5,
    experiment_name='Colab_GPU_Training'
)

print(f"Model trained! MLflow run: {run_id}")
```

**What it does**:

### 1️⃣ Start MLflow Run
```python
mlflow.set_experiment(experiment_name)
with mlflow.start_run() as run:
    # Everything tracked automatically
```

### 2️⃣ Log Parameters
```python
mlflow.log_params(params)  # All XGBoost hyperparameters
mlflow.log_param('threshold', threshold)
mlflow.log_param('n_train', len(X_train))
mlflow.log_param('n_test', len(X_test))
```

### 3️⃣ Train XGBoost Model
```python
model = xgb.XGBClassifier(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=50,
    verbose=False
)
```
**Early stopping**: Stops if no improvement for 50 rounds

### 4️⃣ Make Predictions
```python
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities
y_pred = (y_pred_proba >= threshold).astype(int)  # Binary (0 or 1)
```

### 5️⃣ Calculate Metrics
```python
metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred),
    'auc': roc_auc_score(y_test, y_pred_proba)
}
```

**Metric Meanings**:
- **Accuracy**: Overall correct predictions
- **Precision**: Of predicted churners, how many actually churned
- **Recall**: Of actual churners, how many we caught (most important!)
- **F1**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve (discrimination ability)

### 6️⃣ Log Metrics to MLflow
```python
for metric_name, value in metrics.items():
    mlflow.log_metric(metric_name, value)
```

### 7️⃣ Log Confusion Matrix Values
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
mlflow.log_metric('true_negatives', cm[0, 0])
mlflow.log_metric('false_positives', cm[0, 1])
mlflow.log_metric('false_negatives', cm[1, 0])
mlflow.log_metric('true_positives', cm[1, 1])
```

### 8️⃣ Save Model
```python
mlflow.xgboost.log_model(model, "model")
# Saves model in MLflow format
# Can be loaded later for predictions
```

### 9️⃣ Return Results
```python
return model, run.info.run_id
```

**MLflow UI**:
After training, view results at: `http://localhost:5000`
- Compare multiple runs
- See all metrics side-by-side
- Download best model
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# FILE 3: evaluation.py
# ============================================================================

st.markdown(
    '<div class="file-header">📄 File: src/training/evaluation.py</div>',
    unsafe_allow_html=True,
)

st.markdown("""
**Purpose**: Save best model from MLflow to production directory

**Location**: `src/training/evaluation.py`

**Dependencies**: `mlflow`, `joblib`

**Why**: Export MLflow model to standalone file for deployment
""")

# Function 1
st.markdown('<div class="function-box">', unsafe_allow_html=True)
st.markdown(
    '<div class="function-name">🔹 save_model_from_mlflow()</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
**Description**: Download model from MLflow and save to models/ directory

**Parameters**:
- <span class="param">run_id</span>: `str` - MLflow run ID
- <span class="param">mlflow_tracking_uri</span>: `str` - MLflow tracking URI (default: "./mlruns")
- <span class="param">output_dir</span>: `str` - Where to save model (default: "models/")

**Returns**: <span class="return">str</span> - Path to saved model file

**Use Case**:
```python
# After training, save best model
model_path = save_model_from_mlflow(
    run_id='abc123def456',
    output_dir='models/'
)

print(f"Model saved to: {model_path}")
# Output: models/model_abc123def456.pkl
```

**What it does**:

### 1️⃣ Connect to MLflow
```python
mlflow.set_tracking_uri(mlflow_tracking_uri)
client = MlflowClient()
```

### 2️⃣ Get Run Information
```python
run = client.get_run(run_id)
artifact_uri = f"{run.info.artifact_uri}/model"
```

### 3️⃣ Download Model
```python
model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
```

### 4️⃣ Save as Joblib
```python
output_path = Path(output_dir) / f"model_{run_id}.pkl"
joblib.dump(model, output_path)
```

### 5️⃣ Verify
```python
# Test loading
loaded_model = joblib.load(output_path)
print("✓ Model saved and verified")
```

**Why Save Separately?**
- MLflow great for tracking, but for deployment:
  - Simpler to load from file
  - No MLflow server needed in production
  - Easier to version control
  - Faster loading
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Complete Pipeline Example
st.markdown("""
## 🔄 Complete Training Pipeline

```python
from src.training.optuna_tuning import tune_hyperparameters
from src.training.mlflow_training import train_with_mlflow
from src.training.evaluation import save_model_from_mlflow

# Step 1: Load processed data
train_df = pd.read_csv('data/processed/train_processed.csv')
test_df = pd.read_csv('data/processed/test_processed.csv')

X_train = train_df.drop('churn', axis=1)
y_train = train_df['churn']
X_test = test_df.drop('churn', axis=1)
y_test = test_df['churn']

# Step 2: Hyperparameter Tuning (10-30 min)
print("🔍 Tuning hyperparameters...")
best_params = tune_hyperparameters(
    X_train, y_train,
    X_test, y_test,
    n_trials=100,
    metric='recall'
)
print(f"✓ Best params: {best_params}")

# Step 3: Train with Best Parameters
print("🚂 Training final model...")
model, run_id = train_with_mlflow(
    X_train, y_train,
    X_test, y_test,
    params=best_params,
    threshold=0.5,
    experiment_name='Colab_GPU_Training'
)
print(f"✓ Model trained! Run ID: {run_id}")

# Step 4: Save for Production
print("💾 Saving model...")
model_path = save_model_from_mlflow(
    run_id=run_id,
    output_dir='models/'
)
print(f"✓ Model saved to: {model_path}")

# Step 5: View in MLflow UI
print("📊 View results at: http://localhost:5000")
```

## 📊 MLflow Experiment Tracking

**What Gets Tracked**:
- ✅ All hyperparameters
- ✅ All metrics (accuracy, precision, recall, F1, AUC)
- ✅ Confusion matrix values
- ✅ Model artifact
- ✅ Training duration
- ✅ Dataset sizes

**MLflow UI Features**:
1. **Compare runs** side-by-side
2. **Filter/sort** by any metric
3. **Visualize** metric trends
4. **Download** models
5. **Share** results with team

**Example MLflow Output**:
```
Run: 096c396d6abf4a7da6bae24acb8d99fe
Experiment: Colab_GPU_Training

Parameters:
  max_depth: 6
  learning_rate: 0.08
  n_estimators: 500
  threshold: 0.5

Metrics:
  accuracy: 0.8523
  precision: 0.7891
  recall: 0.9234  ← Most important for churn!
  f1_score: 0.8513
  auc: 0.9156

Artifacts:
  model/
    ├── model.pkl
    ├── conda.yaml
    └── MLmodel
```

## 🎯 Key Takeaways

1. **optuna_tuning.py**: Automatically finds best hyperparameters
   - Tries 100+ combinations
   - Optimizes for recall (catching churners)
   - Saves hours of manual tuning

2. **mlflow_training.py**: Trains model and tracks everything
   - Logs all parameters and metrics
   - Saves model for later use
   - Enables experiment comparison

3. **evaluation.py**: Exports model for production
   - Saves as standalone file
   - No MLflow dependency in production
   - Ready for API deployment

**Training Flow**:
```
Optuna (find best params) 
    → MLflow (train & track) 
        → Evaluation (save for prod) 
            → API Deployment
```

**Next Section**: 🔮 Inference Engine (using trained model for predictions!)
""")

st.markdown("---")
st.markdown("👉 **Navigate to 'Inference Engine' in the sidebar to continue**")
