"""XGBoost training with MLflow using best parameters from Optuna and configurable threshold."""

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
import mlflow
import os

# Force mlflow tracking directory
MLFLOW_DIR = "/content/drive/Othercomputers/My Laptop/machine_learning/E2E_projects/sales-data-churn/mlruns"
os.makedirs(MLFLOW_DIR, exist_ok=True)
mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")

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

    # FILE-BASED tracking (local/CI)
    if mlflow_tracking_uri.startswith("file:"):
        real_path = mlflow_tracking_uri.replace("file:", "")
        os.makedirs(real_path, exist_ok=True)

    # Plain local path (production local)
    elif not mlflow_tracking_uri.startswith(
        ("http:", "https:", "sqlite:", "postgresql:", "mysql:", "databricks")
    ):
        # convert to file uri
        os.makedirs(mlflow_tracking_uri, exist_ok=True)
        mlflow_tracking_uri = f"file:{mlflow_tracking_uri}"

    # Remote server / DB backend
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
    best_params: Dict[str, Any] = None,
) -> Dict[str, Any]:

    # Set MLflow tracking URI
    # setup mlflow safely
    mlflow_tracking_uri = setup_mlflow_tracking(mlflow_tracking_uri)

    # Safe MLflow directory setup (local + CI + prod)
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

    # Run Optuna only if best_params not provided
    if best_params is None:
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
    else:
        print("STEP 1: Using provided best parameters (skipping Optuna optimization)")
        print("\nBest parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")

    # Setup experiment

    print("STEP 2: Training single experiment with best parameters from Optuna")
    print(f"       Using threshold = {threshold_value} for predictions")

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

    # Run single experiment with the best parameters from Optuna
    with mlflow.start_run(run_name=f"best_params_threshold_{threshold_value}") as run:
        # Prepare model parameters
        model_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "tree_method": "hist",
            "device": "cuda",
            "verbosity": 0,
            **best_params,
        }

        # Log parameters
        mlflow.log_param("threshold", threshold_value)
        mlflow.log_param("optuna_trials", n_optuna_trials)

        for param, value in model_params.items():
            mlflow.log_param(param, value)

        # Train model
        model = xgb.XGBClassifier(**model_params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Predict with threshold (using the function argument)
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

        # Save model safely (works local + CI + production)
        os.makedirs(model_save_dir, exist_ok=True)

        safe_threshold = str(threshold_value).replace(".", "_")
        model_pkl_filename = f"model_threshold_{safe_threshold}.pkl"
        model_pkl_path = os.path.join(model_save_dir, model_pkl_filename)

        with open(model_pkl_path, "wb") as f:
            pickle.dump(model, f)

        # Log the pickle file as an artifact
        mlflow.log_artifact(model_pkl_path, artifact_path="models")

        # Store run results (maintaining backward compatibility with pipeline)
        run_result = {
            "run_id": run.info.run_id,
            "run_number": 1,
            "recall": recall,
            "precision": precision,
            "f1_score": f1,
            "accuracy": accuracy,
            "auc": auc,
            "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        }

        # Structure results to maintain backward compatibility
        experiment_results = {
            "experiment_name": experiment_name,
            "experiment_id": experiment_id,
            "threshold": threshold_value,
            "runs": [run_result],
            "best_recall": recall,
            "best_run_id": run.info.run_id,
            "best_metrics": {
                "recall": recall,
                "precision": precision,
                "f1_score": f1,
                "accuracy": accuracy,
                "auc": auc,
            },
        }

        print(
            f"    Recall: {recall:.4f}, Precision: {precision:.4f}, "
            f"F1: {f1:.4f}, AUC: {auc:.4f}"
        )

    return {
        "experiment_results": experiment_results,
        "best_params": best_params,
        "mlflow_tracking_uri": mlflow_tracking_uri,
    }
