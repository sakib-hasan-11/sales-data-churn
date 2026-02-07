"""
Modular Pipeline Runner - Run Individual Stages
=================================================
Run specific stages of the pipeline independently for debugging,
testing, or iterative development.

Usage:
    # Run all stages
    python scripts/modular_pipeline.py --all

    # Run specific stages
    python scripts/modular_pipeline.py --stages load preprocess features

    # Run from a specific stage onwards
    python scripts/modular_pipeline.py --from features

    # Skip certain stages
    python scripts/modular_pipeline.py --skip validation optuna

Available Stages:
    - load: Load raw data
    - validate: Validate data quality
    - preprocess: Clean and preprocess data
    - features: Build engineered features
    - encode: Encode and scale features
    - optuna: Optimize hyperparameters
    - train: Train model with MLflow
    - save: Save best model
"""

import argparse
import pickle
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


class PipelineState:
    """Store and retrieve pipeline state between stages"""

    def __init__(self, state_dir: Path = None):
        self.state_dir = state_dir or (project_root / "outputs" / "pipeline_state")
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def save(self, stage_name: str, data: dict):
        """Save stage output"""
        state_file = self.state_dir / f"{stage_name}.pkl"
        with open(state_file, "wb") as f:
            pickle.dump(data, f)
        print(f"   State saved: {state_file}")

    def load(self, stage_name: str) -> dict:
        """Load stage output"""
        state_file = self.state_dir / f"{stage_name}.pkl"
        if not state_file.exists():
            raise FileNotFoundError(f"State file not found: {state_file}")
        with open(state_file, "rb") as f:
            return pickle.load(f)


def run_load_stage(state: PipelineState, config: dict):
    """Stage: Load raw data"""
    print("\n" + "=" * 70)
    print("STAGE: LOAD DATA")
    print("=" * 70)

    data_dir = project_root / "data" / "raw"

    train_df = load_data(data_dir / "train.csv")
    test_df = load_data(data_dir / "test.csv")

    print(f"\nLoaded: Train={train_df.shape}, Test={test_df.shape}")

    state.save("load", {"train": train_df, "test": test_df})
    return {"train": train_df, "test": test_df}


def run_validate_stage(state: PipelineState, data: dict = None):
    """Stage: Validate data quality"""
    print("\n" + "=" * 70)
    print("STAGE: VALIDATE DATA")
    print("=" * 70)

    if data is None:
        data = state.load("load")

    train_valid, train_failed = validate_data(data["train"])
    test_valid, test_failed = validate_data(data["test"])

    result = {
        "train_valid": train_valid,
        "test_valid": test_valid,
        "train_failed": train_failed,
        "test_failed": test_failed,
    }

    print(
        f"\nValidation: Train={'PASS' if train_valid else 'FAIL'}, Test={'PASS' if test_valid else 'FAIL'}"
    )

    state.save("validate", result)
    return data  # Pass through the data


def run_preprocess_stage(state: PipelineState, data: dict = None):
    """Stage: Preprocess data"""
    print("\n" + "=" * 70)
    print("STAGE: PREPROCESS DATA")
    print("=" * 70)

    if data is None:
        data = state.load("load")

    train_df = raw_preprocess(data["train"], strategy="auto")
    test_df = raw_preprocess(data["test"], strategy="auto")

    print(f"\nPreprocessed: Train={train_df.shape}, Test={test_df.shape}")

    result = {"train": train_df, "test": test_df}
    state.save("preprocess", result)
    return result


def run_features_stage(state: PipelineState, data: dict = None):
    """Stage: Build features"""
    print("\n" + "=" * 70)
    print("STAGE: BUILD FEATURES")
    print("=" * 70)

    if data is None:
        data = state.load("preprocess")

    train_df = build_feature(data["train"])
    test_df = build_feature(data["test"])

    print(f"\nFeatures built: Train={train_df.shape}, Test={test_df.shape}")

    result = {"train": train_df, "test": test_df}
    state.save("features", result)
    return result


def run_encode_stage(state: PipelineState, data: dict = None):
    """Stage: Encode and scale features"""
    print("\n" + "=" * 70)
    print("STAGE: ENCODE & SCALE FEATURES")
    print("=" * 70)

    if data is None:
        data = state.load("features")

    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_df = preprocess_features(data["train"], processed_dir, "train_processed.csv")
    test_df = preprocess_features(data["test"], processed_dir, "test_processed.csv")

    print(f"\nEncoded: Train={train_df.shape}, Test={test_df.shape}")

    result = {"train": train_df, "test": test_df}
    state.save("encode", result)
    return result


def run_optuna_stage(state: PipelineState, data: dict = None, config: dict = None):
    """Stage: Optimize hyperparameters"""
    print("\n" + "=" * 70)
    print("STAGE: HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)

    if data is None:
        data = state.load("encode")

    n_trials = config.get("n_trials", 100) if config else 100
    metric = config.get("metric", "recall") if config else "recall"

    results = optimize_xgboost_hyperparameters(
        train_data=data["train"],
        test_data=data["test"],
        target_col="churn",
        n_trials=n_trials,
        optimize_metric=metric,
    )

    print(f"\nBest {metric}: {results['best_score']:.4f}")

    state.save("optuna", results)
    return results


def run_train_stage(
    state: PipelineState,
    data: dict = None,
    optuna_results: dict = None,
    config: dict = None,
):
    """Stage: Train model with MLflow"""
    print("\n" + "=" * 70)
    print("STAGE: TRAIN MODEL")
    print("=" * 70)

    if data is None:
        data = state.load("encode")

    mlruns_dir = project_root / "mlruns"
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    setup_mlflow_tracking(str(mlruns_dir))

    n_trials = config.get("n_trials", 100) if config else 100
    n_runs = config.get("n_runs", 5) if config else 5
    threshold = config.get("threshold", 0.5) if config else 0.5

    results = train_xgboost_with_mlflow(
        train_data=data["train"],
        test_data=data["test"],
        target_col="churn",
        threshold_value=threshold,
        experiment_name="Modular_Pipeline_Experiment",
        n_optuna_trials=n_trials,
        n_runs=n_runs,
        mlflow_tracking_uri=str(mlruns_dir),
        model_save_dir=str(model_dir),
    )

    best_recall = results["experiment_results"]["best_recall"]
    best_run_id = results["experiment_results"]["best_run_id"]

    print(f"\nBest recall: {best_recall:.4f}")
    print(f"Run ID: {best_run_id}")

    state.save("train", results)
    return results


def run_save_stage(state: PipelineState, train_results: dict = None):
    """Stage: Save best model"""
    print("\n" + "=" * 70)
    print("STAGE: SAVE MODEL")
    print("=" * 70)

    if train_results is None:
        train_results = state.load("train")

    mlruns_dir = project_root / "mlruns"
    model_dir = project_root / "models"

    best_run_id = train_results["experiment_results"]["best_run_id"]

    model_info = save_model_from_run(
        run_id=best_run_id,
        mlflow_tracking_uri=str(mlruns_dir),
        model_save_dir=str(model_dir),
    )

    print(f"\nModel saved: {model_info['model_path']}")
    print(f"Recall: {model_info['recall']:.4f}")

    state.save("save", model_info)
    return model_info


def main():
    """Main execution with argument parsing"""

    parser = argparse.ArgumentParser(description="Modular ML Pipeline Runner")
    parser.add_argument("--all", action="store_true", help="Run all stages")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=[
            "load",
            "validate",
            "preprocess",
            "features",
            "encode",
            "optuna",
            "train",
            "save",
        ],
        help="Specific stages to run",
    )
    parser.add_argument(
        "--from",
        dest="from_stage",
        choices=[
            "load",
            "validate",
            "preprocess",
            "features",
            "encode",
            "optuna",
            "train",
            "save",
        ],
        help="Run from this stage onwards",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=[
            "load",
            "validate",
            "preprocess",
            "features",
            "encode",
            "optuna",
            "train",
            "save",
        ],
        help="Stages to skip",
    )
    parser.add_argument(
        "--n-trials", type=int, default=100, help="Number of Optuna trials"
    )
    parser.add_argument("--n-runs", type=int, default=5, help="Number of MLflow runs")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Classification threshold"
    )
    parser.add_argument(
        "--metric",
        default="recall",
        choices=["recall", "precision", "f1", "auc"],
        help="Optimization metric",
    )

    args = parser.parse_args()

    # Determine which stages to run
    all_stages = [
        "load",
        "validate",
        "preprocess",
        "features",
        "encode",
        "optuna",
        "train",
        "save",
    ]

    if args.all:
        stages_to_run = all_stages
    elif args.stages:
        stages_to_run = args.stages
    elif args.from_stage:
        start_idx = all_stages.index(args.from_stage)
        stages_to_run = all_stages[start_idx:]
    else:
        # Default: run all
        stages_to_run = all_stages

    # Remove skipped stages
    if args.skip:
        stages_to_run = [s for s in stages_to_run if s not in args.skip]

    # Configuration
    config = {
        "n_trials": args.n_trials,
        "n_runs": args.n_runs,
        "threshold": args.threshold,
        "metric": args.metric,
    }

    # Initialize state manager
    state = PipelineState()

    print("\n" + "=" * 70)
    print("MODULAR PIPELINE RUNNER")
    print("=" * 70)
    print(f"\nStages to run: {', '.join(stages_to_run)}")
    if args.skip:
        print(f"Stages to skip: {', '.join(args.skip)}")
    print()

    # Run stages
    try:
        data = None
        optuna_results = None
        train_results = None

        for stage in stages_to_run:
            if stage == "load":
                data = run_load_stage(state, config)
            elif stage == "validate":
                data = run_validate_stage(state, data)
            elif stage == "preprocess":
                data = run_preprocess_stage(state, data)
            elif stage == "features":
                data = run_features_stage(state, data)
            elif stage == "encode":
                data = run_encode_stage(state, data)
            elif stage == "optuna":
                optuna_results = run_optuna_stage(state, data, config)
            elif stage == "train":
                train_results = run_train_stage(state, data, optuna_results, config)
            elif stage == "save":
                run_save_stage(state, train_results)

        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ PIPELINE FAILED!")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
