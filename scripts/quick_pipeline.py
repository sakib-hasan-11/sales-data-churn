"""
Quick Pipeline Runner - Minimal Version
========================================
A simplified pipeline script for quick experimentation and testing.

This script runs a streamlined version of the pipeline without extensive
logging and configuration options. Use this for rapid prototyping.

Usage:
    python scripts/quick_pipeline.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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


def quick_pipeline():
    """Run a quick version of the pipeline with minimal configuration"""

    print("🚀 Starting Quick Pipeline...\n")

    # Configuration
    DATA_DIR = project_root / "data" / "raw"
    PROCESSED_DIR = project_root / "data" / "processed"
    MODEL_DIR = project_root / "models"
    MLRUNS_DIR = project_root / "mlruns"

    # Create directories
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print("📥 Loading data...")
    train_df = load_data(DATA_DIR / "train.csv")
    test_df = load_data(DATA_DIR / "test.csv")
    print(f"   Train: {train_df.shape}, Test: {test_df.shape}")

    # 2. Validate (optional - can comment out for speed)
    print("\n✅ Validating data...")
    train_valid, _ = validate_data(train_df)
    test_valid, _ = validate_data(test_df)
    if train_valid and test_valid:
        print("   Validation passed!")
    else:
        print("   ⚠️  Some validation checks failed, but continuing...")

    # 3. Preprocess
    print("\n🧹 Preprocessing data...")
    train_df = raw_preprocess(train_df, strategy="auto")
    test_df = raw_preprocess(test_df, strategy="auto")
    print(f"   Missing values handled")

    # 4. Build Features
    print("\n🔧 Building features...")
    train_df = build_feature(train_df)
    test_df = build_feature(test_df)
    print(f"   Features created: {train_df.shape[1]} columns")

    # 5. Preprocess Features
    print("\n📊 Processing features...")
    train_df = preprocess_features(train_df, PROCESSED_DIR, "train_processed.csv")
    test_df = preprocess_features(test_df, PROCESSED_DIR, "test_processed.csv")
    print(f"   Final shape - Train: {train_df.shape}, Test: {test_df.shape}")

    # 6. Optimize Hyperparameters
    print("\n⚙️  Optimizing hyperparameters (this may take a while)...")
    optuna_results = optimize_xgboost_hyperparameters(
        train_data=train_df,
        test_data=test_df,
        target_col="churn",
        n_trials=50,  # Reduced for speed
        optimize_metric="recall",
    )
    print(f"   Best recall: {optuna_results['best_score']:.4f}")

    # 7. Train with MLflow
    print("\n🤖 Training model with MLflow...")
    setup_mlflow_tracking(str(MLRUNS_DIR))

    training_results = train_xgboost_with_mlflow(
        train_data=train_df,
        test_data=test_df,
        target_col="churn",
        threshold_value=0.5,
        experiment_name="Quick_Pipeline_Experiment",
        n_optuna_trials=50,  # Reduced for speed
        n_runs=3,  # Reduced for speed
        mlflow_tracking_uri=str(MLRUNS_DIR),
        model_save_dir=str(MODEL_DIR),
    )

    best_recall = training_results["experiment_results"]["best_recall"]
    best_run_id = training_results["experiment_results"]["best_run_id"]
    print(f"   Best recall: {best_recall:.4f}")
    print(f"   Run ID: {best_run_id}")

    # 8. Save Model
    print("\n💾 Saving best model...")
    model_info = save_model_from_run(
        run_id=best_run_id,
        mlflow_tracking_uri=str(MLRUNS_DIR),
        model_save_dir=str(MODEL_DIR),
    )

    print(f"\n✅ Pipeline Complete!")
    print(f"   Model saved: {model_info['model_path']}")
    print(f"   Final recall: {model_info['recall']:.4f}")
    print(f"\n💡 View MLflow UI: mlflow ui --backend-store-uri {MLRUNS_DIR}")

    return model_info


if __name__ == "__main__":
    try:
        quick_pipeline()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
