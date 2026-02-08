# Google Colab GPU Training Setup Guide

## 🚀 Quick Start (3 Steps)

### Step 1: Enable GPU in Colab
```
Runtime > Change runtime type > Hardware accelerator > GPU
```

### Step 2: Clone Your Repository
In a Colab notebook cell:
```python
# Clone your project
!git clone https://github.com/your-username/sales-data-churn.git
%cd sales-data-churn

# Or upload manually if not on GitHub
from google.colab import files
# Upload your project as zip, then:
# !unzip sales-data-churn.zip
# %cd sales-data-churn
```

### Step 3: Install Dependencies & Run
```python
# Install required packages
!pip install -q xgboost scikit-learn pandas numpy mlflow optuna

# Run the GPU-optimized pipeline
!python scripts/colab_pipeline.py
```

---

## 📊 What This Pipeline Does

✅ **Skips Optuna** - Uses pre-computed optimal hyperparameters (saves time & GPU credits)  
✅ **GPU Training** - Utilizes Colab's CUDA GPU for faster XGBoost training  
✅ **Single Run** - Trains one optimized model (not multiple experiments)  
✅ **900 Trees** - XGBoost with n_estimators=900 for maximum performance  
✅ **Recall Optimized** - Tuned for catching churners (threshold=0.35)  

---

## 📁 File Structure Required

Make sure your project has this structure:
```
sales-data-churn/
├── data/
│   └── raw/
│       ├── train.csv    # Your training data
│       ├── test.csv     # Your test data
│       └── holdout.csv  # (optional)
├── src/
│   ├── data_processing/
│   ├── features/
│   ├── training/
│   └── utils/
└── scripts/
    └── colab_pipeline.py   # The GPU pipeline
```

---

## ⚙️ Pre-configured Hyperparameters

The pipeline uses these optimal parameters (found via 250 Optuna trials):
```python
{
    "booster": "gbtree",
    "max_depth": 7,
    "eta": 0.296,
    "n_estimators": 900,
    "subsample": 0.76,
    "colsample_bytree": 0.99,
    "colsample_bylevel": 0.78,
    "min_child_weight": 6,
    "lambda": 0.00032,
    "alpha": 0.00017,
    "gamma": 0.00017
}
```

---

## 📥 Download Your Trained Model

After training completes:
```python
from google.colab import files
files.download('models/best_model.pkl')
```

---

## 🔍 Verify GPU is Working

```python
# Quick GPU check
!nvidia-smi

# Or in Python
import xgboost as xgb
try:
    xgb.XGBClassifier(tree_method='hist', device='cuda')
    print("✓ GPU is available")
except:
    print("⚠ GPU not available")
```

---

## ⏱️ Expected Training Time

With Colab GPU (T4):
- **Without Optuna:** ~2-5 minutes
- **With Optuna (250 trials):** ~30-60 minutes

Our pipeline skips Optuna, so expect **~2-5 minutes total runtime**!

---

## 📊 Monitoring Training

The pipeline will display:
- Data loading progress
- Feature engineering steps
- Training metrics (recall, precision, F1)
- MLflow experiment tracking
- Model save location

---

## 🎯 Customization Options

Edit `scripts/colab_pipeline.py` to change:

```python
# In ColabPipelineConfig class:
THRESHOLD_VALUE = 0.35        # Decision threshold (lower = catch more churners)
OPTIMIZE_METRIC = "recall"    # Metric to optimize
PREPROCESSING_STRATEGY = "median"  # Handling missing values
```

To run Optuna optimization (takes longer):
```python
# Set BEST_PARAMS = None in config
# The pipeline will automatically run Optuna
```

---

## 🐛 Troubleshooting

### GPU Not Detected
- Check Runtime > Change runtime type > GPU is selected
- Restart runtime: Runtime > Restart runtime
- Verify with `!nvidia-smi`

### Module Not Found
```python
# Make sure you're in the right directory
%cd /content/sales-data-churn

# Reinstall packages
!pip install --upgrade xgboost scikit-learn mlflow
```

### Out of Memory
- Reduce `n_estimators` from 900 to 500
- Use smaller dataset for testing
- Restart runtime to clear memory

---

## 📚 Additional Resources

- [XGBoost GPU Support](https://xgboost.readthedocs.io/en/latest/gpu/index.html)
- [Google Colab GPU Guide](https://colab.research.google.com/notebooks/gpu.ipynb)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)

---

## ✨ Tips for Colab

1. **Save work periodically** - Colab sessions timeout after inactivity
2. **Connect to Drive** - Store results in Google Drive for persistence
3. **Use GPU wisely** - Free tier has usage limits
4. **Download models** - Before session ends

---

Happy Training! 🎉
