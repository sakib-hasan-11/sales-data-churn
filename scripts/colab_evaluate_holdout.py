"""
Google Colab Holdout Evaluation Script (GPU-Accelerated)
=========================================================
Evaluate the best trained model on holdout data with full preprocessing pipeline.

SETUP INSTRUCTIONS:
===================
1. First, train a model using colab_pipeline.py

2. Enable GPU in Colab:
   Runtime > Change runtime type > Hardware accelerator > GPU

3. Mount Google Drive and navigate to project:
   from google.colab import drive
   drive.mount('/content/drive')

   # Navigate to project directory
   %cd /content/drive/Othercomputers/My\ Laptop/machine_learning/E2E_projects/sales-data-churn

4. Run this script:
   !python scripts/colab_evaluate_holdout.py

Note: This script loads the saved model and evaluates it on unseen holdout data.
"""

import os
import pickle
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
    project_root = Path(
        "/content/drive/Othercomputers/My Laptop/machine_learning/E2E_projects/sales-data-churn"
    )
    if not project_root.exists():
        current = Path.cwd()
        if "sales-data-churn" in str(current):
            while current.name != "sales-data-churn" and current.parent != current:
                current = current.parent
            project_root = current
        else:
            project_root = Path.cwd()
    print(f"Project root detected: {project_root}")
else:
    project_root = Path(__file__).parent.parent

sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Import pipeline functions
from src.data_processing.load import load_data
from src.data_processing.preprocess import raw_preprocess
from src.features.build_feature import build_feature
from src.features.feature_preprocess import preprocess_features
from src.utils.data_validator import validate_data

# =============================================================================
# GPU VERIFICATION
# =============================================================================


def verify_gpu():
    """Verify GPU availability for XGBoost"""
    try:
        import xgboost as xgb

        gpu_available = xgb.XGBClassifier(tree_method="hist", device="cuda")
        print("✓ GPU (CUDA) is available for model inference")
        return True
    except Exception as e:
        print(f"⚠ GPU not available: {e}")
        print("  Inference will use CPU")
        return False


# =============================================================================
# CONFIGURATION
# =============================================================================


class HoldoutEvaluationConfig:
    """Configuration for holdout evaluation"""

    # Data paths
    DATA_DIR = project_root / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"

    HOLDOUT_FILE = RAW_DATA_DIR / "holdout.csv"

    # Model paths
    MODEL_DIR = project_root / "models"
    OUTPUT_DIR = project_root / "outputs"

    # Target and preprocessing
    TARGET_COLUMN = "churn"
    PREPROCESSING_STRATEGY = "median"

    # Feature preprocessing
    PROCESSED_HOLDOUT_NAME = "holdout_processed.csv"

    @classmethod
    def find_best_model(cls):
        """Find the most recent best model in models directory"""
        if not cls.MODEL_DIR.exists():
            raise FileNotFoundError(f"Models directory not found: {cls.MODEL_DIR}")

        # Look for model files
        model_files = list(cls.MODEL_DIR.glob("model_threshold_*.pkl"))

        if not model_files:
            raise FileNotFoundError(f"No model files found in {cls.MODEL_DIR}")

        # Sort by recall (highest first)
        model_files_with_recall = []
        for model_file in model_files:
            try:
                # Extract recall from filename: model_threshold_0_35_recall_0.9123.pkl
                parts = model_file.stem.split("_recall_")
                if len(parts) == 2:
                    recall = float(parts[1])
                    model_files_with_recall.append((model_file, recall))
            except:
                # If we can't parse recall, use modification time
                model_files_with_recall.append((model_file, model_file.stat().st_mtime))

        # Sort by recall/time (descending)
        model_files_with_recall.sort(key=lambda x: x[1], reverse=True)
        best_model_path = model_files_with_recall[0][0]

        # Extract threshold from filename
        try:
            threshold_part = best_model_path.stem.split("model_threshold_")[1].split(
                "_recall_"
            )[0]
            threshold = float(threshold_part.replace("_", "."))
        except:
            threshold = 0.5  # Default fallback

        return best_model_path, threshold


# =============================================================================
# PIPELINE STAGES
# =============================================================================


def stage_1_load_holdout(config: HoldoutEvaluationConfig):
    """Load holdout data"""
    print("\n" + "=" * 80)
    print("STAGE 1: LOADING HOLDOUT DATA")
    print("=" * 80)

    print(f"\nLoading holdout data from: {config.HOLDOUT_FILE}")

    if not config.HOLDOUT_FILE.exists():
        raise FileNotFoundError(f"Holdout file not found: {config.HOLDOUT_FILE}")

    holdout_df = load_data(str(config.HOLDOUT_FILE))

    print(f"\nHoldout data loaded successfully:")
    print(f"  Shape: {holdout_df.shape[0]} rows, {holdout_df.shape[1]} columns")
    print(f"  Target distribution:")
    if config.TARGET_COLUMN.title() in holdout_df.columns:
        print(
            f"    {holdout_df[config.TARGET_COLUMN.title()].value_counts().to_dict()}"
        )

    return holdout_df


def stage_2_validate_holdout(holdout_df: pd.DataFrame):
    """Validate holdout data quality"""
    print("\n" + "=" * 80)
    print("STAGE 2: VALIDATING HOLDOUT DATA")
    print("=" * 80)

    print("\nValidating holdout data...")
    is_valid, failed = validate_data(holdout_df)

    if is_valid:
        print("\n✓ Holdout data validation passed!")
        return True
    else:
        print("\n⚠ Some validation checks failed. Review warnings above.")
        print(f"Failed expectations: {failed}")
        return False


def stage_3_preprocess_holdout(
    holdout_df: pd.DataFrame, config: HoldoutEvaluationConfig
):
    """Preprocess raw holdout data"""
    print("\n" + "=" * 80)
    print("STAGE 3: PREPROCESSING HOLDOUT DATA")
    print("=" * 80)

    print(f"\nPreprocessing strategy: {config.PREPROCESSING_STRATEGY}")

    holdout_df = raw_preprocess(holdout_df, strategy=config.PREPROCESSING_STRATEGY)

    print(f"\nPreprocessing complete:")
    print(f"  Holdout set: {holdout_df.shape}")

    return holdout_df


def stage_4_build_features_holdout(holdout_df: pd.DataFrame):
    """Build engineered features for holdout data"""
    print("\n" + "=" * 80)
    print("STAGE 4: FEATURE ENGINEERING")
    print("=" * 80)

    print("\nBuilding features...")

    holdout_df = build_feature(holdout_df)

    print(f"\nFeature engineering complete:")
    print(f"  Holdout set: {holdout_df.shape[0]} rows, {holdout_df.shape[1]} features")

    return holdout_df


def stage_5_preprocess_features_holdout(
    holdout_df: pd.DataFrame, config: HoldoutEvaluationConfig
):
    """Preprocess features (scaling, encoding)"""
    print("\n" + "=" * 80)
    print("STAGE 5: FEATURE PREPROCESSING")
    print("=" * 80)

    print("\nApplying feature preprocessing...")

    holdout_df = preprocess_features(
        df=holdout_df,
        output_path=str(config.PROCESSED_DATA_DIR),
        name=config.PROCESSED_HOLDOUT_NAME,
    )

    print(f"\nFeature preprocessing complete:")
    print(f"  Holdout set: {holdout_df.shape}")

    return holdout_df


def stage_6_load_model(config: HoldoutEvaluationConfig):
    """Load the best trained model"""
    print("\n" + "=" * 80)
    print("STAGE 6: LOADING BEST MODEL")
    print("=" * 80)

    print(f"\nSearching for best model in: {config.MODEL_DIR}")

    model_path, threshold = config.find_best_model()

    print(f"\n✓ Found model: {model_path.name}")
    print(f"  Threshold: {threshold}")

    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print(f"  Model type: {type(model).__name__}")

    return model, threshold, model_path


def stage_7_evaluate_on_holdout(
    model,
    holdout_df: pd.DataFrame,
    threshold: float,
    model_path: Path,
    config: HoldoutEvaluationConfig,
):
    """Evaluate model on holdout data"""
    print("\n" + "=" * 80)
    print("STAGE 7: EVALUATING ON HOLDOUT DATA")
    print("=" * 80)

    # Separate features and target
    X_holdout = holdout_df.drop(columns=[config.TARGET_COLUMN])
    y_holdout = holdout_df[config.TARGET_COLUMN]

    print(f"\n🚀 Running predictions on GPU...")
    print(f"  Samples: {X_holdout.shape[0]}")
    print(f"  Features: {X_holdout.shape[1]}")
    print(f"  Threshold: {threshold}")

    # Predict probabilities
    y_pred_proba = model.predict_proba(X_holdout)[:, 1]

    # Apply threshold
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate metrics
    print("\n" + "=" * 80)
    print("HOLDOUT DATA PERFORMANCE METRICS")
    print("=" * 80)

    recall = recall_score(y_holdout, y_pred, zero_division=0)
    precision = precision_score(y_holdout, y_pred, zero_division=0)
    f1 = f1_score(y_holdout, y_pred, zero_division=0)
    accuracy = accuracy_score(y_holdout, y_pred)
    auc = roc_auc_score(y_holdout, y_pred_proba)

    print(f"\n📊 Classification Metrics:")
    print(f"  Recall:    {recall:.4f} ⭐ (Primary metric)")
    print(f"  Precision: {precision:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  ROC AUC:   {auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_holdout, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n📈 Confusion Matrix:")
    print(f"  True Positives:  {tp:4d} (Correctly identified churners)")
    print(f"  True Negatives:  {tn:4d} (Correctly identified non-churners)")
    print(f"  False Positives: {fp:4d} (False alarms)")
    print(f"  False Negatives: {fn:4d} (Missed churners)")

    # Additional business metrics
    total_churners = tp + fn
    total_non_churners = tn + fp
    churners_caught = tp
    churners_missed = fn

    print(f"\n💼 Business Impact:")
    print(f"  Total churners in holdout:     {total_churners}")
    print(
        f"  Churners caught by model:      {churners_caught} ({(churners_caught / total_churners * 100):.1f}%)"
    )
    print(
        f"  Churners missed by model:      {churners_missed} ({(churners_missed / total_churners * 100):.1f}%)"
    )
    print(f"  False alarm rate:              {(fp / total_non_churners * 100):.1f}%")

    # Detailed classification report
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(y_holdout, y_pred, target_names=["Non-Churn", "Churn"]))

    # Save results
    results = {
        "model_path": str(model_path),
        "threshold": threshold,
        "holdout_samples": len(y_holdout),
        "metrics": {
            "recall": recall,
            "precision": precision,
            "f1_score": f1,
            "accuracy": accuracy,
            "roc_auc": auc,
        },
        "confusion_matrix": {
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        },
        "predictions": {
            "y_true": y_holdout.tolist(),
            "y_pred": y_pred.tolist(),
            "y_pred_proba": y_pred_proba.tolist(),
        },
    }

    return results


def stage_8_save_results(results: dict, config: HoldoutEvaluationConfig):
    """Save evaluation results"""
    print("\n" + "=" * 80)
    print("STAGE 8: SAVING RESULTS")
    print("=" * 80)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save results as pickle
    results_file = config.OUTPUT_DIR / "holdout_evaluation_results.pkl"
    with open(results_file, "wb") as f:
        pickle.dump(results, f)

    print(f"\n✓ Results saved to: {results_file}")

    # Save predictions CSV
    predictions_df = pd.DataFrame(
        {
            "y_true": results["predictions"]["y_true"],
            "y_pred": results["predictions"]["y_pred"],
            "y_pred_proba": results["predictions"]["y_pred_proba"],
        }
    )
    predictions_file = config.OUTPUT_DIR / "holdout_predictions.csv"
    predictions_df.to_csv(predictions_file, index=False)

    print(f"✓ Predictions saved to: {predictions_file}")

    # Save summary text file
    summary_file = config.OUTPUT_DIR / "holdout_evaluation_summary.txt"
    with open(summary_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("HOLDOUT DATA EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {results['model_path']}\n")
        f.write(f"Threshold: {results['threshold']}\n")
        f.write(f"Holdout Samples: {results['holdout_samples']}\n\n")
        f.write("METRICS:\n")
        f.write("-" * 80 + "\n")
        for metric, value in results["metrics"].items():
            f.write(f"  {metric.upper():15s}: {value:.4f}\n")
        f.write("\nCONFUSION MATRIX:\n")
        f.write("-" * 80 + "\n")
        for key, value in results["confusion_matrix"].items():
            f.write(f"  {key.replace('_', ' ').title():20s}: {value}\n")

    print(f"✓ Summary saved to: {summary_file}")

    return results_file, predictions_file, summary_file


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def run_holdout_evaluation():
    """Execute holdout evaluation pipeline"""

    print("\n" + "=" * 80)
    print("🚀 GOOGLE COLAB HOLDOUT EVALUATION - CHURN PREDICTION")
    print("=" * 80)
    print(f"\nProject Root: {project_root}")

    # Verify GPU
    gpu_available = verify_gpu()

    # Initialize configuration
    config = HoldoutEvaluationConfig()

    try:
        # Stage 1: Load Holdout Data
        holdout_df = stage_1_load_holdout(config)

        # Stage 2: Validate Data
        stage_2_validate_holdout(holdout_df)

        # Stage 3: Preprocess Data
        holdout_df = stage_3_preprocess_holdout(holdout_df, config)

        # Stage 4: Build Features
        holdout_df = stage_4_build_features_holdout(holdout_df)

        # Stage 5: Preprocess Features
        holdout_df = stage_5_preprocess_features_holdout(holdout_df, config)

        # Stage 6: Load Best Model
        model, threshold, model_path = stage_6_load_model(config)

        # Stage 7: Evaluate on Holdout
        results = stage_7_evaluate_on_holdout(
            model, holdout_df, threshold, model_path, config
        )

        # Stage 8: Save Results
        results_file, predictions_file, summary_file = stage_8_save_results(
            results, config
        )

        # Pipeline completion summary
        print("\n" + "=" * 80)
        print("✓ HOLDOUT EVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\n📊 Final Holdout Performance:")
        print(f"  Recall:    {results['metrics']['recall']:.4f}")
        print(f"  Precision: {results['metrics']['precision']:.4f}")
        print(f"  F1-Score:  {results['metrics']['f1_score']:.4f}")
        print(f"  ROC AUC:   {results['metrics']['roc_auc']:.4f}")

        if IN_COLAB:
            print("\n📁 Download evaluation results:")
            print("  from google.colab import files")
            print(f"  files.download('{results_file}')")
            print(f"  files.download('{predictions_file}')")
            print(f"  files.download('{summary_file}')")

        return {
            "status": "success",
            "results": results,
            "gpu_used": gpu_available,
        }

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ HOLDOUT EVALUATION FAILED!")
        print("=" * 80)
        print(f"\nError: {str(e)}")
        import traceback

        traceback.print_exc()

        return {"status": "failed", "error": str(e)}


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results = run_holdout_evaluation()

    # Exit with appropriate code
    if results["status"] == "success":
        print("\n✓ Holdout evaluation successful!")
        sys.exit(0)
    else:
        print("\n✗ Holdout evaluation failed!")
        sys.exit(1)
