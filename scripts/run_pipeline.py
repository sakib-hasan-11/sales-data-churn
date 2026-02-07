"""
Complete ML Pipeline for Sales Data Churn Prediction
======================================================
This script orchestrates the entire machine learning pipeline from data loading
to model deployment, using all functions from the src/ folder.

Pipeline Stages:
1. Data Loading
2. Data Validation
3. Data Preprocessing
4. Feature Engineering
5. Feature Preprocessing
6. Hyperparameter Tuning
7. Model Training & Tracking
8. Model Evaluation & Saving

Usage:
    python scripts/run_pipeline.py
"""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

# Import all pipeline functions
from src.data_processing.load import load_data
from src.data_processing.preprocess import raw_preprocess
from src.features.build_feature import build_feature
from src.features.feature_preprocess import preprocess_features
from src.training.evaluation import save_model_from_run
from src.training.mlflow_training import (
    setup_mlflow_tracking,
    train_xgboost_with_mlflow,
)
from src.training.optuna_tuning import optimize_xgboost_hyperparameters
from src.utils.data_validator import validate_data

# =============================================================================
# CONFIGURATION
# =============================================================================


class PipelineConfig:
    """Pipeline configuration settings"""

    # Data paths
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

    # Model training parameters
    TARGET_COLUMN = "churn"
    THRESHOLD_VALUE = 0.5
    N_OPTUNA_TRIALS = 100
    N_MLFLOW_RUNS = 5
    OPTIMIZE_METRIC = "recall"  # Options: 'recall', 'precision', 'f1', 'auc'

    # MLflow settings
    EXPERIMENT_NAME = "Churn_Prediction_Pipeline"
    MLFLOW_TRACKING_URI = str(MLRUNS_DIR)

    # Data preprocessing strategy
    PREPROCESSING_STRATEGY = "auto"  # Options: 'auto', 'median', 'mode', 'constant'
    CUSTOM_FILL_VALUES = None  # Optional: {"column_name": fill_value}

    # Feature preprocessing
    PROCESSED_TRAIN_NAME = "train_processed.csv"
    PROCESSED_TEST_NAME = "test_processed.csv"

    @classmethod
    def setup_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.MLRUNS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# PIPELINE STAGES
# =============================================================================


def stage_1_load_data(config: PipelineConfig):
    """
    Stage 1: Load raw data from CSV files

    Returns:
        tuple: (train_df, test_df)
    """
    print("\n" + "=" * 80)
    print("STAGE 1: DATA LOADING")
    print("=" * 80)

    print(f"\nLoading training data from: {config.TRAIN_FILE}")
    train_df = load_data(config.TRAIN_FILE)

    print(f"\nLoading test data from: {config.TEST_FILE}")
    test_df = load_data(config.TEST_FILE)

    print(f"\nData Loading Summary:")
    print(f"  Training samples: {len(train_df):,}")
    print(f"  Test samples: {len(test_df):,}")
    print(f"  Features: {len(train_df.columns)}")

    return train_df, test_df


def stage_2_validate_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Stage 2: Validate data quality using Great Expectations

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame

    Returns:
        bool: True if validation passes, False otherwise
    """
    print("\n" + "=" * 80)
    print("STAGE 2: DATA VALIDATION")
    print("=" * 80)

    print("\nValidating training data...")
    train_success, train_failed = validate_data(train_df)

    print("\nValidating test data...")
    test_success, test_failed = validate_data(test_df)

    print(f"\nValidation Results:")
    print(f"  Training data: {'PASSED' if train_success else 'FAILED'}")
    if not train_success:
        print(f"    Failed checks: {train_failed}")

    print(f"  Test data: {'PASSED' if test_success else 'FAILED'}")
    if not test_success:
        print(f"    Failed checks: {test_failed}")

    if not (train_success and test_success):
        print("\nWARNING: Data validation failed. Proceeding with caution...")
        return False

    print("\nAll validation checks passed!")
    return True


def stage_3_preprocess_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame, config: PipelineConfig
):
    """
    Stage 3: Clean column names and handle missing values

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        config: Pipeline configuration

    Returns:
        tuple: (preprocessed_train_df, preprocessed_test_df)
    """
    print("\n" + "=" * 80)
    print("STAGE 3: DATA PREPROCESSING")
    print("=" * 80)

    print("\nPreprocessing training data...")
    train_preprocessed = raw_preprocess(
        train_df,
        strategy=config.PREPROCESSING_STRATEGY,
        custom_fill=config.CUSTOM_FILL_VALUES,
    )

    print("\nPreprocessing test data...")
    test_preprocessed = raw_preprocess(
        test_df,
        strategy=config.PREPROCESSING_STRATEGY,
        custom_fill=config.CUSTOM_FILL_VALUES,
    )

    print(f"\nPreprocessing Summary:")
    print(f"  Training shape: {train_preprocessed.shape}")
    print(f"  Test shape: {test_preprocessed.shape}")
    print(f"  Missing values in train: {train_preprocessed.isnull().sum().sum()}")
    print(f"  Missing values in test: {test_preprocessed.isnull().sum().sum()}")

    return train_preprocessed, test_preprocessed


def stage_4_build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Stage 4: Create engineered features

    Args:
        train_df: Preprocessed training DataFrame
        test_df: Preprocessed test DataFrame

    Returns:
        tuple: (train_with_features, test_with_features)
    """
    print("\n" + "=" * 80)
    print("STAGE 4: FEATURE ENGINEERING")
    print("=" * 80)

    print("\nBuilding features for training data...")
    train_features = build_feature(train_df)

    print("\nBuilding features for test data...")
    test_features = build_feature(test_df)

    print(f"\nFeature Engineering Summary:")
    print(f"  Training shape: {train_features.shape}")
    print(f"  Test shape: {test_features.shape}")
    print(f"  Total features: {len(train_features.columns)}")

    # Show newly created features
    original_cols = set(train_df.columns)
    new_cols = [col for col in train_features.columns if col not in original_cols]
    print(f"\nNew features created ({len(new_cols)}):")
    for col in new_cols:
        print(f"    - {col}")

    return train_features, test_features


def stage_5_preprocess_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, config: PipelineConfig
):
    """
    Stage 5: Encode categories and scale numerical features

    Args:
        train_df: Training DataFrame with engineered features
        test_df: Test DataFrame with engineered features
        config: Pipeline configuration

    Returns:
        tuple: (train_processed, test_processed)
    """
    print("\n" + "=" * 80)
    print("STAGE 5: FEATURE PREPROCESSING")
    print("=" * 80)

    print("\nEncoding and scaling training features...")
    train_processed = preprocess_features(
        train_df,
        output_path=config.PROCESSED_DATA_DIR,
        name=config.PROCESSED_TRAIN_NAME,
    )

    print("\nEncoding and scaling test features...")
    test_processed = preprocess_features(
        test_df, output_path=config.PROCESSED_DATA_DIR, name=config.PROCESSED_TEST_NAME
    )

    print(f"\nFeature Preprocessing Summary:")
    print(f"  Training shape: {train_processed.shape}")
    print(f"  Test shape: {test_processed.shape}")
    print(f"  Processed data saved to: {config.PROCESSED_DATA_DIR}")

    return train_processed, test_processed


def stage_6_optimize_hyperparameters(
    train_df: pd.DataFrame, test_df: pd.DataFrame, config: PipelineConfig
):
    """
    Stage 6: Optimize XGBoost hyperparameters using Optuna

    Args:
        train_df: Processed training DataFrame
        test_df: Processed test DataFrame
        config: Pipeline configuration

    Returns:
        dict: Optimization results with best parameters
    """
    print("\n" + "=" * 80)
    print("STAGE 6: HYPERPARAMETER TUNING")
    print("=" * 80)

    print(f"\nOptimizing hyperparameters with Optuna...")
    print(f"  Optimization metric: {config.OPTIMIZE_METRIC}")
    print(f"  Number of trials: {config.N_OPTUNA_TRIALS}")

    results = optimize_xgboost_hyperparameters(
        train_data=train_df,
        test_data=test_df,
        target_col=config.TARGET_COLUMN,
        n_trials=config.N_OPTUNA_TRIALS,
        optimize_metric=config.OPTIMIZE_METRIC,
    )

    print(f"\nOptimization Results:")
    print(f"  Best {config.OPTIMIZE_METRIC}: {results['best_score']:.4f}")
    print(f"\nBest Hyperparameters:")
    for param, value in results["best_params"].items():
        print(f"    {param}: {value}")

    return results


def stage_7_train_with_mlflow(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_params: dict,
    config: PipelineConfig,
):
    """
    Stage 7: Train XGBoost models with MLflow tracking

    Args:
        train_df: Processed training DataFrame
        test_df: Processed test DataFrame
        best_params: Best hyperparameters from Optuna
        config: Pipeline configuration

    Returns:
        dict: Training results with MLflow run information
    """
    print("\n" + "=" * 80)
    print("STAGE 7: MODEL TRAINING & TRACKING")
    print("=" * 80)

    # Setup MLflow tracking
    print(f"\nSetting up MLflow tracking...")
    print(f"  Tracking URI: {config.MLFLOW_TRACKING_URI}")
    print(f"  Experiment name: {config.EXPERIMENT_NAME}")

    setup_mlflow_tracking(config.MLFLOW_TRACKING_URI)

    # Train model with MLflow
    print(f"\nTraining XGBoost model...")
    print(f"  Threshold: {config.THRESHOLD_VALUE}")
    print(f"  Number of runs: {config.N_MLFLOW_RUNS}")

    training_results = train_xgboost_with_mlflow(
        train_data=train_df,
        test_data=test_df,
        target_col=config.TARGET_COLUMN,
        threshold_value=config.THRESHOLD_VALUE,
        experiment_name=config.EXPERIMENT_NAME,
        n_optuna_trials=config.N_OPTUNA_TRIALS,
        n_runs=config.N_MLFLOW_RUNS,
        mlflow_tracking_uri=config.MLFLOW_TRACKING_URI,
        model_save_dir=str(config.MODEL_DIR),
        best_params=best_params,
    )

    print(f"\nTraining Results:")
    print(f"  Best recall: {training_results['experiment_results']['best_recall']:.4f}")
    print(f"  Best run ID: {training_results['experiment_results']['best_run_id']}")

    return training_results


def stage_8_save_best_model(training_results: dict, config: PipelineConfig):
    """
    Stage 8: Save the best model to production folder

    Args:
        training_results: Results from MLflow training
        config: Pipeline configuration

    Returns:
        dict: Model save information
    """
    print("\n" + "=" * 80)
    print("STAGE 8: MODEL EVALUATION & SAVING")
    print("=" * 80)

    best_run_id = training_results["experiment_results"]["best_run_id"]

    print(f"\nSaving best model to production...")
    print(f"  Run ID: {best_run_id}")
    print(f"  Destination: {config.MODEL_DIR}")

    model_info = save_model_from_run(
        run_id=best_run_id,
        mlflow_tracking_uri=config.MLFLOW_TRACKING_URI,
        model_save_dir=str(config.MODEL_DIR),
    )

    print(f"\nModel Saved Successfully:")
    print(f"  Path: {model_info['model_path']}")
    print(f"  Recall: {model_info['recall']:.4f}")
    print(f"  Threshold: {model_info['threshold']}")

    return model_info


# =============================================================================
# MAIN PIPELINE EXECUTION
# =============================================================================


def run_pipeline():
    """Execute the complete ML pipeline"""

    print("\n" + "=" * 80)
    print("SALES DATA CHURN PREDICTION - ML PIPELINE")
    print("=" * 80)
    print(f"\nProject Root: {project_root}")

    # Initialize configuration
    config = PipelineConfig()
    config.setup_directories()

    try:
        # Stage 1: Load Data
        train_df, test_df = stage_1_load_data(config)

        # Stage 2: Validate Data
        validation_passed = stage_2_validate_data(train_df, test_df)

        # Stage 3: Preprocess Data
        train_df, test_df = stage_3_preprocess_data(train_df, test_df, config)

        # Stage 4: Build Features
        train_df, test_df = stage_4_build_features(train_df, test_df)

        # Stage 5: Preprocess Features
        train_df, test_df = stage_5_preprocess_features(train_df, test_df, config)

        # Stage 6: Optimize Hyperparameters (SKIPPED - Using pre-computed best parameters)
        # optuna_results = stage_6_optimize_hyperparameters(train_df, test_df, config)

        # Using pre-computed best hyperparameters from previous Optuna optimization
        print("\n" + "=" * 80)
        print("STAGE 6: USING PRE-COMPUTED HYPERPARAMETERS (Skipping Optuna)")
        print("=" * 80)

        best_params = {
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

        print("\nUsing Best Hyperparameters:")
        for param, value in best_params.items():
            print(f"    {param}: {value}")

        # Stage 7: Train with MLflow
        training_results = stage_7_train_with_mlflow(
            train_df, test_df, best_params, config
        )

        # Stage 8: Save Best Model
        model_info = stage_8_save_best_model(training_results, config)

        # Pipeline completion summary
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nFinal Model Performance:")
        print(f"  Recall: {model_info['recall']:.4f}")
        print(f"  Threshold: {model_info['threshold']}")
        print(f"  Model Path: {model_info['model_path']}")
        print(f"\nMLflow Tracking URI: {config.MLFLOW_TRACKING_URI}")
        print(
            f"View experiments: mlflow ui --backend-store-uri {config.MLFLOW_TRACKING_URI}"
        )

        return {
            "status": "success",
            "model_info": model_info,
            "training_results": training_results,
            "best_params": best_params,
        }

    except Exception as e:
        print("\n" + "=" * 80)
        print("PIPELINE FAILED!")
        print("=" * 80)
        print(f"\nError: {str(e)}")
        import traceback

        traceback.print_exc()

        return {"status": "failed", "error": str(e)}


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results = run_pipeline()

    # Exit with appropriate code
    if results["status"] == "success":
        sys.exit(0)
    else:
        sys.exit(1)
