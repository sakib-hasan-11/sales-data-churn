"""
Google Colab Pipeline for Sales Data Churn Prediction (GPU-Accelerated)
========================================================================
This script is optimized for running in Google Colab with GPU support.

SETUP INSTRUCTIONS:
===================
1. Enable GPU in Colab:
   Runtime > Change runtime type > Hardware accelerator > GPU

2. Mount Google Drive and navigate to project:
   from google.colab import drive
   drive.mount('/content/drive')

   # Navigate to project directory
   %cd /content/drive/Othercomputers/My\ Laptop/machine_learning/E2E_projects/sales-data-churn

3. Install required packages:
   !pip install -q xgboost scikit-learn pandas numpy mlflow optuna

4. Run this script:
   !python scripts/colab_pipeline.py

Note: The script automatically detects the correct project path from Google Drive.
All outputs (models, mlruns, processed data) will be saved to your Google Drive.

GPU Training with Pre-computed Optimal Hyperparameters
=======================================================
"""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Colab-specific setup
try:
    import google.colab

    IN_COLAB = True
    print("✓ Running in Google Colab")
except ImportError:
    IN_COLAB = False
    print("✓ Running locally")

# Add project root to path
if IN_COLAB:
    # Google Drive mount path
    project_root = Path(
        "/content/drive/Othercomputers/My Laptop/machine_learning/E2E_projects/sales-data-churn"
    )
    if not project_root.exists():
        # Fallback: try to detect from current working directory
        current = Path.cwd()
        if "sales-data-churn" in str(current):
            # Navigate up to project root
            while current.name != "sales-data-churn" and current.parent != current:
                current = current.parent
            project_root = current
        else:
            project_root = Path.cwd()
    print(f"Project root detected: {project_root}")
else:
    project_root = Path(__file__).parent.parent

sys.path.insert(0, str(project_root))

import pandas as pd

# Import pipeline functions
from src.data_processing.load import load_data
from src.data_processing.preprocess import raw_preprocess
from src.features.build_feature import build_feature
from src.features.feature_preprocess import preprocess_features
from src.training.evaluation import save_model_from_run
from src.training.mlflow_training import (
    setup_mlflow_tracking,
    train_xgboost_with_mlflow,
)
from src.utils.data_validator import validate_data

# =============================================================================
# GPU VERIFICATION
# =============================================================================


def verify_gpu():
    """Verify GPU availability for XGBoost"""
    try:
        import xgboost as xgb

        # Test GPU availability
        gpu_available = xgb.XGBClassifier(tree_method="hist", device="cuda")
        print("✓ GPU (CUDA) is available for XGBoost training")
        return True
    except Exception as e:
        print(f"⚠ GPU not available: {e}")
        print("  Training will fall back to CPU")
        return False


# =============================================================================
# COLAB CONFIGURATION
# =============================================================================


class ColabPipelineConfig:
    """Pipeline configuration optimized for Colab"""

    # Data paths (adjust if needed)
    DATA_DIR = project_root / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"

    TRAIN_FILE = RAW_DATA_DIR / "train.csv"
    TEST_FILE = RAW_DATA_DIR / "test.csv"
    HOLDOUT_FILE = RAW_DATA_DIR / "holdout.csv"

    # Output paths
    OUTPUT_DIR = project_root / "outputs"
    MODEL_DIR = project_root / "models"
    MLRUNS_DIR = project_root / "mlruns"

    # Model training parameters (production settings)
    TARGET_COLUMN = "churn"
    THRESHOLD_VALUE = 0.35  # Lower threshold for better recall
    OPTIMIZE_METRIC = "recall"

    # MLflow settings
    EXPERIMENT_NAME = "Colab_GPU_Training"
    MLFLOW_TRACKING_URI = str(MLRUNS_DIR)

    # Data preprocessing
    PREPROCESSING_STRATEGY = "median"

    # Feature preprocessing
    PROCESSED_TRAIN_NAME = "train_processed.csv"
    PROCESSED_TEST_NAME = "test_processed.csv"

    # Pre-computed best hyperparameters from Optuna optimization
    # These were found to be optimal and will be used directly
    BEST_PARAMS = {
        "booster": "gbtree",
        "lambda": 0.00032762263951052436,
        "alpha": 0.00017370640229832804,
        "max_depth": 7,
        "eta": 0.2960673713462837,
        "gamma": 0.00017131007397068948,
        "min_child_weight": 6,
        "subsample": 0.7605678991335877,
        "colsample_bytree": 0.9988324896159033,
        "colsample_bylevel": 0.7777131466076425,
        "n_estimators": 900,
    }

    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        print("\nCreating project directories...")

        cls.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Processed data: {cls.PROCESSED_DATA_DIR}")

        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Outputs: {cls.OUTPUT_DIR}")

        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Models: {cls.MODEL_DIR}")

        cls.MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ MLflow runs: {cls.MLRUNS_DIR}")

        # Verify mlruns directory is created and accessible
        if cls.MLRUNS_DIR.exists() and cls.MLRUNS_DIR.is_dir():
            print(f"\n✓ MLruns directory confirmed at: {cls.MLRUNS_DIR}")
        else:
            print(f"\n⚠ Warning: Could not verify mlruns directory")


# =============================================================================
# PIPELINE STAGES
# =============================================================================


def stage_1_load_data(config: ColabPipelineConfig):
    """Load training and test data"""
    print("\n" + "=" * 80)
    print("STAGE 1: DATA LOADING")
    print("=" * 80)

    print(f"\nLoading data from: {config.RAW_DATA_DIR}")

    train_df = load_data(str(config.TRAIN_FILE))
    test_df = load_data(str(config.TEST_FILE))

    print(f"\nData loaded successfully:")
    print(f"  Training set: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    print(f"  Test set: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

    return train_df, test_df


def stage_2_validate_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Validate data quality"""
    print("\n" + "=" * 80)
    print("STAGE 2: DATA VALIDATION")
    print("=" * 80)

    print("\nValidating training data...")
    train_is_valid, train_failed = validate_data(train_df)

    print("\nValidating test data...")
    test_is_valid, test_failed = validate_data(test_df)

    if train_is_valid and test_is_valid:
        print("\n✓ All data validation checks passed!")
        return True
    else:
        print("\n⚠ Some validation checks failed. Review warnings above.")
        return False


def stage_3_preprocess_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame, config: ColabPipelineConfig
):
    """Preprocess raw data"""
    print("\n" + "=" * 80)
    print("STAGE 3: DATA PREPROCESSING")
    print("=" * 80)

    print(f"\nPreprocessing strategy: {config.PREPROCESSING_STRATEGY}")

    train_df = raw_preprocess(train_df, strategy=config.PREPROCESSING_STRATEGY)
    test_df = raw_preprocess(test_df, strategy=config.PREPROCESSING_STRATEGY)

    print(f"\nPreprocessing complete:")
    print(f"  Training set: {train_df.shape}")
    print(f"  Test set: {test_df.shape}")

    return train_df, test_df


def stage_4_build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Build engineered features"""
    print("\n" + "=" * 80)
    print("STAGE 4: FEATURE ENGINEERING")
    print("=" * 80)

    print("\nBuilding features...")

    train_df = build_feature(train_df)
    test_df = build_feature(test_df)

    print(f"\nFeature engineering complete:")
    print(f"  Training set: {train_df.shape[0]} rows, {train_df.shape[1]} features")
    print(f"  Test set: {test_df.shape[0]} rows, {test_df.shape[1]} features")

    return train_df, test_df


def stage_5_preprocess_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, config: ColabPipelineConfig
):
    """Preprocess features (scaling, encoding)"""
    print("\n" + "=" * 80)
    print("STAGE 5: FEATURE PREPROCESSING")
    print("=" * 80)

    print("\nApplying feature preprocessing...")

    print("\nProcessing training data...")
    train_df = preprocess_features(
        df=train_df,
        output_path=str(config.PROCESSED_DATA_DIR),
        name=config.PROCESSED_TRAIN_NAME,
    )

    print("\nProcessing test data...")
    test_df = preprocess_features(
        df=test_df,
        output_path=str(config.PROCESSED_DATA_DIR),
        name=config.PROCESSED_TEST_NAME,
    )

    print(f"\nFeature preprocessing complete:")
    print(f"  Training set: {train_df.shape}")
    print(f"  Test set: {test_df.shape}")

    return train_df, test_df


def stage_6_train_with_mlflow_gpu(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ColabPipelineConfig,
):
    """Train XGBoost model on GPU with pre-computed best parameters"""
    print("\n" + "=" * 80)
    print("STAGE 6: GPU-ACCELERATED MODEL TRAINING")
    print("=" * 80)

    # Setup MLflow tracking
    print("\nSetting up MLflow tracking...")
    print(f"  Tracking URI: {config.MLFLOW_TRACKING_URI}")
    print(f"  Experiment name: {config.EXPERIMENT_NAME}")

    # Ensure mlruns directory exists before MLflow setup
    config.MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

    setup_mlflow_tracking(config.MLFLOW_TRACKING_URI)

    # Display pre-computed parameters
    print("\n✓ Using pre-computed optimal hyperparameters (skipping Optuna):")
    for param, value in config.BEST_PARAMS.items():
        print(f"    {param}: {value}")

    # Train model with GPU
    print(f"\n🚀 Training XGBoost model on GPU...")
    print(f"  Threshold: {config.THRESHOLD_VALUE}")
    print(f"  Device: CUDA (GPU)")
    print(f"  Boosting rounds: {config.BEST_PARAMS['n_estimators']}")

    training_results = train_xgboost_with_mlflow(
        train_data=train_df,
        test_data=test_df,
        target_col=config.TARGET_COLUMN,
        threshold_value=config.THRESHOLD_VALUE,
        experiment_name=config.EXPERIMENT_NAME,
        n_optuna_trials=0,  # Not used (we provide best_params)
        n_runs=1,
        mlflow_tracking_uri=config.MLFLOW_TRACKING_URI,
        model_save_dir=str(config.MODEL_DIR),
        best_params=config.BEST_PARAMS,  # Use pre-computed parameters
    )

    print("\n✓ Training complete!")
    print(f"  Best recall: {training_results['experiment_results']['best_recall']:.4f}")
    print(f"  Run ID: {training_results['experiment_results']['best_run_id']}")

    return training_results


def stage_7_save_best_model(training_results: dict, config: ColabPipelineConfig):
    """Save the best model"""
    print("\n" + "=" * 80)
    print("STAGE 7: SAVING BEST MODEL")
    print("=" * 80)

    print("\nSaving best model to production folder...")

    best_run_id = training_results["experiment_results"]["best_run_id"]

    model_info = save_model_from_run(
        run_id=best_run_id,
        mlflow_tracking_uri=config.MLFLOW_TRACKING_URI,
        model_save_dir=str(config.MODEL_DIR),
    )

    print("\n✓ Model saved successfully!")
    print(f"  Recall: {model_info['recall']:.4f}")
    print(f"  Threshold: {model_info['threshold']}")
    print(f"  Model Path: {model_info['model_path']}")

    return model_info


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def run_colab_pipeline():
    """Execute the complete ML pipeline in Colab with GPU"""

    print("\n" + "=" * 80)
    print("🚀 GOOGLE COLAB GPU PIPELINE - CHURN PREDICTION")
    print("=" * 80)
    print(f"\nProject Root: {project_root}")

    # Verify GPU
    gpu_available = verify_gpu()

    # Initialize configuration
    config = ColabPipelineConfig()
    config.setup_directories()

    try:
        # Stage 1: Load Data
        train_df, test_df = stage_1_load_data(config)

        # Stage 2: Validate Data
        stage_2_validate_data(train_df, test_df)

        # Stage 3: Preprocess Data
        train_df, test_df = stage_3_preprocess_data(train_df, test_df, config)

        # Stage 4: Build Features
        train_df, test_df = stage_4_build_features(train_df, test_df)

        # Stage 5: Preprocess Features
        train_df, test_df = stage_5_preprocess_features(train_df, test_df, config)

        # Stage 6: Train with GPU (using pre-computed parameters)
        training_results = stage_6_train_with_mlflow_gpu(train_df, test_df, config)

        # Stage 7: Save Best Model
        model_info = stage_7_save_best_model(training_results, config)

        # Pipeline completion summary
        print("\n" + "=" * 80)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\n📊 Final Model Performance:")
        print(f"  Recall: {model_info['recall']:.4f}")
        print(f"  Threshold: {model_info['threshold']}")
        print(f"  Model Path: {model_info['model_path']}")

        if IN_COLAB:
            print("\n📁 Download your trained model:")
            print("  from google.colab import files")
            print(f"  files.download('{model_info['model_path']}')")

        print("\n📈 View MLflow Experiments:")
        print(f"  MLflow Tracking URI: {config.MLFLOW_TRACKING_URI}")

        return {
            "status": "success",
            "model_info": model_info,
            "training_results": training_results,
            "best_params": config.BEST_PARAMS,
            "gpu_used": gpu_available,
        }

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ PIPELINE FAILED!")
        print("=" * 80)
        print(f"\nError: {str(e)}")
        import traceback

        traceback.print_exc()

        return {"status": "failed", "error": str(e)}


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results = run_colab_pipeline()

    # Exit with appropriate code
    if results["status"] == "success":
        print("\n✓ Pipeline execution successful!")
        sys.exit(0)
    else:
        print("\n✗ Pipeline execution failed!")
        sys.exit(1)
