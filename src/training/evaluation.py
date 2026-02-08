"""Save model from MLflow run to production folder."""

import os
import pickle

import mlflow
from mlflow.tracking import MlflowClient


def save_model_from_run(
    run_id: str,
    mlflow_tracking_uri: str = "./mlruns",
    model_save_dir: str = "./models",
):
    """
    Save model from MLflow run to production folder.

    Parameters
    ----------
    run_id : str
        MLflow run ID
    mlflow_tracking_uri : str
        Path to mlruns folder (default: "./mlruns")
    model_save_dir : str
        Where to save the model (default: "./models")

    Returns
    -------
    dict
        Model path and metrics
    """

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    os.makedirs(model_save_dir, exist_ok=True)

    print(f"Loading model from run: {run_id}")

    # Get run info
    client = MlflowClient()
    run = client.get_run(run_id)

    # Get metrics
    recall = run.data.metrics.get("recall", 0.0)
    threshold = run.data.params.get("threshold", "unknown")

    print(f"Recall: {recall:.4f}, Threshold: {threshold}")

    # Format threshold to match saved filename (dots replaced with underscores)
    safe_threshold = str(threshold).replace(".", "_")

    # Load model (try multiple paths)
    model = None
    try:
        # Try pickle artifact first (with safe threshold formatting)
        artifact_uri = f"runs:/{run_id}/models/model_threshold_{safe_threshold}.pkl"
        local_path = mlflow.artifacts.download_artifacts(artifact_uri)
        with open(local_path, "rb") as f:
            model = pickle.load(f)
        print("✓ Loaded from pickle artifact")
    except Exception as e:
        print(f"  ⚠ Could not load from pickle artifact: {e}")
        try:
            # Try MLflow format
            model = mlflow.xgboost.load_model(
                f"runs:/{run_id}/model_threshold_{safe_threshold}"
            )
            print("✓ Loaded from MLflow format")
        except Exception as e2:
            print(f"  ⚠ Could not load from MLflow format: {e2}")
            # Fallback to generic path
            model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
            print("✓ Loaded from generic path")

    # Save to production folder
    model_filename = f"model_threshold_{threshold}_recall_{recall:.4f}.pkl"
    model_path = os.path.join(model_save_dir, model_filename)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"✓ Model saved to: {model_path}")

    return {
        "model_path": model_path,
        "recall": recall,
        "threshold": threshold,
        "run_id": run_id,
    }


# Example usage
if __name__ == "__main__":
    # Replace with your actual run_id from MLflow UI
    result = save_model_from_run(
        run_id="your_run_id_here",
        mlflow_tracking_uri="./mlruns",
        model_save_dir="./models",
    )

    print(f"\nDone! Model saved to: {result['model_path']}")
