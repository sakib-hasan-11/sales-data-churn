"""XGBoost training with MLflow experiments for different threshold values."""

import os
import pickle
from typing import Any, Dict

import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .optuna_tuning import optimize_xgboost_hyperparameters


# ===== CI SAFE MODE =====
if os.getenv("CI") == "true":
    os.makedirs("mlruns", exist_ok=True)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Modular_Pipeline_Experiment")


def setup_mlflow_tracking(mlflow_tracking_uri: str):
    """
    Production-safe MLflow setup.
    Supports:
    - local dev
    - pytest temp dir
    - GitHub Actions
    - docker
    - real production mlruns
    - remote MLflow server
    """

    if not mlflow_tracking_uri:
        # default local production folder
        mlflow_tracking_uri = "file:./mlruns"

    # -------------------------------
    # FILE-BASED tracking (local/CI)
    # -------------------------------
    if mlflow_tracking_uri.startswith("file:"):
        real_path = mlflow_tracking_uri.replace("file:", "")
        os.makedirs(real_path, exist_ok=True)

    # -------------------------------
    # Plain local path (production local)
    # -------------------------------
    elif not mlflow_tracking_uri.startswith(
        ("http:", "https:", "sqlite:", "postgresql:", "mysql:", "databricks")
    ):
        # convert to file uri
        os.makedirs(mlflow_tracking_uri, exist_ok=True)
        mlflow_tracking_uri = f"file:{mlflow_tracking_uri}"

    # -------------------------------
    # Remote server / DB backend
    # -------------------------------
    else:
        # do nothing (server manages storage)
        pass

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    return mlflow_tracking_uri


def train_xgboost_with_mlflow(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    target_col: str,
    threshold_value: float,
    experiment_name: str = "XGBoost_Threshold_Experiment",
    n_optuna_trials: int = 100,
    n_runs: int = 5,
    mlflow_tracking_uri: str = "./mlruns",
    model_save_dir="models",
) -> Dict[str, Any]:
    """
    Train XGBoost model with MLflow experiment for a single threshold value.

    Creates one experiment with multiple runs using the best parameters from Optuna.
    Each run uses slightly different random seeds for robustness testing.

    Parameters
    ----------
    train_data : pd.DataFrame
        Training dataset with features and target column
    test_data : pd.DataFrame
        Test dataset with features and target column
    target_col : str
        Name of the target column
    threshold_value : float
        Threshold value for binary classification (e.g., 0.5)
    experiment_name : str, optional
        Name for the MLflow experiment (default: "XGBoost_Threshold_Experiment")
    n_optuna_trials : int, optional
        Number of Optuna trials for hyperparameter optimization (default: 100)
    n_runs : int, optional
        Number of runs for this threshold value (default: 5)
    mlflow_tracking_uri : str, optional
        MLflow tracking URI (default: "./mlruns")

    Returns
    -------
    Dict[str, Any]
        Dictionary containing experiment results and best run information
    """

    # Set MLflow tracking URI
    # setup mlflow safely
    mlflow_tracking_uri = setup_mlflow_tracking(mlflow_tracking_uri)

    # --------------------------------------------------
    # Safe MLflow directory setup (local + CI + prod)
    # --------------------------------------------------
    if mlflow_tracking_uri.startswith("file:"):
        real_path = mlflow_tracking_uri.replace("file:", "")
        os.makedirs(real_path, exist_ok=True)

    elif mlflow_tracking_uri.startswith(("http:", "https:", "databricks", "sqlite:")):
        # remote tracking server → do nothing
        pass

    else:
        # plain local path
        os.makedirs(mlflow_tracking_uri, exist_ok=True)
        mlflow_tracking_uri = f"file:{mlflow_tracking_uri}"

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Separate features and target
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]

    # First, run Optuna to get best hyperparameters

    print("STEP 1: Running Optuna hyperparameter optimization...")

    optuna_results = optimize_xgboost_hyperparameters(
        train_data=train_data,
        test_data=test_data,
        target_col=target_col,
        n_trials=n_optuna_trials,
        optimize_metric="recall",
    )

    best_params = optuna_results["best_params"]

    print("\nBest parameters found by Optuna:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    # Setup experiment

    print(f"STEP 2: Training with threshold = {threshold_value}")

    # Create or get experiment
    try:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=f"{mlflow_tracking_uri}/{experiment_name}",
        )
    except Exception:
        # Experiment already exists
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)

    # Store runs for this threshold
    experiment_results = {
        "experiment_name": experiment_name,
        "experiment_id": experiment_id,
        "threshold": threshold_value,
        "runs": [],
        "best_recall": 0.0,
        "best_run_id": None,
        "best_metrics": {},
    }

    # Run multiple experiments with the best parameters
    for run_num in range(1, n_runs + 1):
        print(f"\n  Run {run_num}/{n_runs} for threshold {threshold_value}...")

        with mlflow.start_run(
            run_name=f"run_{run_num}_threshold_{threshold_value}"
        ) as run:
            # Prepare model parameters
            model_params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "random_state": 42 + run_num,  # Vary seed for each run
                "tree_method": "hist",
                "verbosity": 0,
                **best_params,
            }

            # Log parameters
            mlflow.log_param("threshold", threshold_value)
            mlflow.log_param("run_number", run_num)
            mlflow.log_param("optuna_trials", n_optuna_trials)

            for param, value in model_params.items():
                mlflow.log_param(param, value)

            # Train model
            model = xgb.XGBClassifier(**model_params)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

            # Predict with threshold
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= threshold_value).astype(int)

            # Calculate metrics
            recall = recall_score(y_test, y_pred, zero_division=0)
            precision = precision_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)

            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            # Log metrics
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("auc", auc)
            mlflow.log_metric("true_positives", tp)
            mlflow.log_metric("true_negatives", tn)
            mlflow.log_metric("false_positives", fp)
            mlflow.log_metric("false_negatives", fn)

            # Log additional derived metrics
            if (tp + fp) > 0:
                mlflow.log_metric("positive_predictions", tp + fp)
            if (tn + fn) > 0:
                mlflow.log_metric("negative_predictions", tn + fn)

            # --------------------------------------------------
            # Save model safely (works local + CI + production)
            # --------------------------------------------------
            os.makedirs(model_save_dir, exist_ok=True)

            safe_threshold = str(threshold_value).replace(".", "_")
            model_pkl_filename = f"model_threshold_{safe_threshold}.pkl"
            model_pkl_path = os.path.join(model_save_dir, model_pkl_filename)

            with open(model_pkl_path, "wb") as f:
                pickle.dump(model, f)

            # Log the pickle file as an artifact
            mlflow.log_artifact(model_pkl_path, artifact_path="models")

            # Also log model using MLflow's built-in method
            safe_threshold = str(threshold_value).replace(".", "_")

            mlflow.log_artifact(model_pkl_path, artifact_path="models")

            # Store run results
            run_result = {
                "run_id": run.info.run_id,
                "run_number": run_num,
                "recall": recall,
                "precision": precision,
                "f1_score": f1,
                "accuracy": accuracy,
                "auc": auc,
                "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
            }
            experiment_results["runs"].append(run_result)

            # Update best recall
            if recall > experiment_results["best_recall"]:
                experiment_results["best_recall"] = recall
                experiment_results["best_run_id"] = run.info.run_id
                experiment_results["best_metrics"] = {
                    "recall": recall,
                    "precision": precision,
                    "f1_score": f1,
                    "accuracy": accuracy,
                    "auc": auc,
                }

            print(
                f"    Recall: {recall:.4f}, Precision: {precision:.4f}, "
                f"F1: {f1:.4f}, AUC: {auc:.4f}"
            )

    # Print summary

    print("EXPERIMENT SUMMARY")

    print(f"\nExperiment Name: {experiment_name}")
    print(f"Experiment ID: {experiment_id}")
    print(f"Threshold: {threshold_value}")
    print(f"Total Runs: {n_runs}")
    print(f"\nBest Recall: {experiment_results['best_recall']:.4f}")
    print(f"Best Run ID: {experiment_results['best_run_id']}")
    print("\nBest Run Metrics:")
    for metric, value in experiment_results["best_metrics"].items():
        print(f"  {metric}: {value:.4f}")

    return {
        "experiment_results": experiment_results,
        "best_params": best_params,
        "mlflow_tracking_uri": mlflow_tracking_uri,
    }


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification

    # Create sample data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42
    )

    # Create dataframes
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y

    # Split data
    train_df = df.iloc[:700]
    test_df = df.iloc[700:]

    # Train with MLflow for a single threshold
    results = train_xgboost_with_mlflow(
        train_data=train_df,
        test_data=test_df,
        target_col="target",
        threshold_value=0.5,
        experiment_name="XGBoost_Churn_Prediction_Threshold_0.5",
        n_optuna_trials=50,  # Reduced for demo
        n_runs=3,  # Reduced for demo
        mlflow_tracking_uri="./mlruns",
    )

    print("EXPERIMENT COMPLETE!")
    print(f"MLflow tracking URI: {results['mlflow_tracking_uri']}")
    print("View results with: mlflow ui")
