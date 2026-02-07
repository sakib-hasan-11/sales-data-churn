# Pipeline Scripts

This folder contains complete ML pipeline scripts that orchestrate all functions from the `src/` folder.

## Available Scripts

### 1. `run_pipeline.py` - Full Production Pipeline ⭐

The complete, production-ready pipeline with extensive logging, error handling, and configuration.

**Features:**
- 8-stage comprehensive pipeline
- Detailed logging and progress tracking
- Error handling and validation
- MLflow experiment tracking
- Configurable parameters

**Usage:**
```bash
python scripts/run_pipeline.py
```

**Pipeline Stages:**
1. Data Loading
2. Data Validation (Great Expectations)
3. Data Preprocessing
4. Feature Engineering
5. Feature Preprocessing
6. Hyperparameter Tuning (Optuna)
7. Model Training & Tracking (MLflow)
8. Model Evaluation & Saving

**Configuration:**
Edit the `PipelineConfig` class in the script to customize:
- File paths
- Number of Optuna trials
- Number of MLflow runs
- Optimization metric
- Threshold value
- Preprocessing strategy

---

### 2. `quick_pipeline.py` - Fast Execution

A streamlined pipeline for rapid prototyping and testing.

**Features:**
- Minimal logging
- Reduced number of trials for speed
- Quick execution
- Same functionality as full pipeline

**Usage:**
```bash
python scripts/quick_pipeline.py
```

**When to Use:**
- Quick experiments
- Testing changes
- Rapid iteration
- Demo purposes

---

### 3. `modular_pipeline.py` - Flexible Stage Execution

Run specific stages of the pipeline independently for debugging and iterative development.

**Features:**
- Run individual stages
- Skip stages
- Start from any stage
- Save/load intermediate results
- Command-line arguments

**Usage:**

```bash
# Run all stages
python scripts/modular_pipeline.py --all

# Run specific stages only
python scripts/modular_pipeline.py --stages load preprocess features

# Run from a specific stage onwards
python scripts/modular_pipeline.py --from features

# Skip certain stages
python scripts/modular_pipeline.py --skip validate optuna

# Customize parameters
python scripts/modular_pipeline.py --all --n-trials 50 --n-runs 3 --threshold 0.6 --metric f1
```

**Command-line Arguments:**
- `--all`: Run all stages
- `--stages`: Specify which stages to run
- `--from`: Run from a specific stage onwards
- `--skip`: Skip specific stages
- `--n-trials`: Number of Optuna trials (default: 100)
- `--n-runs`: Number of MLflow runs (default: 5)
- `--threshold`: Classification threshold (default: 0.5)
- `--metric`: Optimization metric (recall/precision/f1/auc, default: recall)

**Available Stages:**
- `load`: Load raw data
- `validate`: Validate data quality
- `preprocess`: Clean and preprocess data
- `features`: Build engineered features
- `encode`: Encode and scale features
- `optuna`: Optimize hyperparameters
- `train`: Train model with MLflow
- `save`: Save best model

---

### 4. `pipeline_config.yaml` - Configuration File

YAML configuration file for customizing pipeline behavior (for future use).

---

## Output Structure

After running any pipeline script, you'll have:

```
.
├── data/
│   └── processed/
│       ├── train_processed.csv
│       └── test_processed.csv
├── models/
│   └── model_threshold_0.5_recall_0.XXXX.pkl
├── mlruns/
│   └── [MLflow experiment data]
└── outputs/
    └── pipeline_state/  (modular_pipeline only)
        ├── load.pkl
        ├── preprocess.pkl
        ├── features.pkl
        └── ...
```

## Viewing Results

### MLflow UI

To view experiment results:

```bash
mlflow ui --backend-store-uri mlruns
```

Then open: http://localhost:5000

### Check Model Performance

After pipeline completion, check:
- Model file in `models/` directory
- MLflow experiments in MLflow UI
- Processed data in `data/processed/`

---

## Common Workflows

### 1. First Time Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python scripts/run_pipeline.py
```

### 2. Quick Experiment
```bash
# Fast iteration with reduced trials
python scripts/quick_pipeline.py
```

### 3. Debug Specific Stage
```bash
# Run only feature engineering
python scripts/modular_pipeline.py --stages features

# Run from preprocessing onwards
python scripts/modular_pipeline.py --from preprocess
```

### 4. Optimize for Precision Instead of Recall
```bash
# Using modular pipeline
python scripts/modular_pipeline.py --all --metric precision

# Or edit PipelineConfig in run_pipeline.py
# OPTIMIZE_METRIC = "precision"
```

### 5. Skip Validation for Speed
```bash
python scripts/modular_pipeline.py --all --skip validate
```

---

## Troubleshooting

### Issue: FileNotFoundError
**Solution:** Ensure data files exist in `data/raw/`:
- `train.csv`
- `test.csv`
- `holdout.csv` (optional)

### Issue: Import Errors
**Solution:** Run from project root:
```bash
cd /path/to/sales-data-churn
python scripts/run_pipeline.py
```

### Issue: Out of Memory
**Solution:** Reduce trials/runs:
```bash
python scripts/modular_pipeline.py --all --n-trials 20 --n-runs 2
```

### Issue: MLflow Errors
**Solution:** Delete and recreate mlruns:
```bash
rm -rf mlruns
python scripts/run_pipeline.py
```

---

## Pipeline Functions Map

| Stage | Module | Function |
|-------|--------|----------|
| Load Data | `src.data_processing.load` | `load_data()` |
| Validate | `src.utils.data_validator` | `validate_data()` |
| Preprocess | `src.data_processing.preprocess` | `raw_preprocess()` |
| Features | `src.features.build_feature` | `build_feature()` |
| Encode | `src.features.feature_preprocess` | `preprocess_features()` |
| Optuna | `src.training.optuna_tuning` | `optimize_xgboost_hyperparameters()` |
| MLflow Setup | `src.training.mlflow_training` | `setup_mlflow_tracking()` |
| Train | `src.training.mlflow_training` | `train_xgboost_with_mlflow()` |
| Save | `src.training.evaluation` | `save_model_from_run()` |

---

## Tips

1. **Start with Quick Pipeline:** Test everything works before running full pipeline
2. **Use Modular for Development:** Debug individual stages during development
3. **Monitor MLflow:** Keep MLflow UI open to track experiments in real-time
4. **Save Intermediate Results:** Use modular pipeline to save time on reruns
5. **Adjust Trials:** Start with fewer trials (20-50) during development, increase for production

---

## Need Help?

- Check the main project README
- Review function documentation in `src/` folder
- See the interactive app: `streamlit run app.py`
- Check MLflow UI for experiment details
