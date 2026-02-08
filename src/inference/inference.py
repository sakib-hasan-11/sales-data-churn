"""
Production-grade inference module for churn prediction.
Handles model loading, preprocessing, and predictions.
"""

import logging
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processing.preprocess import raw_preprocess
from features.build_feature import build_feature

logger = logging.getLogger(__name__)


# ============================================================================
# PREPROCESSING PIPELINE
# ============================================================================


class InferencePreprocessor:
    """
    Complete preprocessing pipeline for inference.
    Ensures data is transformed consistently with training.
    """

    def __init__(self):
        """Initialize preprocessor."""
        self.feature_names = None
# clean and standardize column names to match training format
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize column names to match training format."""

        def _clean(col: str) -> str:
            col = col.strip()
            col = col.replace("/", "_")
            col = col.replace(" ", "_")
            col = re.sub(r"[^0-9a-zA-Z_]+", "", col)
            col = re.sub(r"_+", "_", col)
            return col.lower()

        new_cols = {c: _clean(c) for c in df.columns}
        return df.rename(columns=new_cols)
    



# encode categorical features using same logic as training
    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        # Label encode gender
        gender_mapping = {"male": 0, "female": 1}
        df["gender"] = (
            df["gender"].str.lower().map(gender_mapping).fillna(0).astype(int)
        )

        # One-hot encoding for categorical features
        subscription_dummies = pd.get_dummies(df["subscription_type"], prefix="sub")
        contract_dummies = pd.get_dummies(df["contract_length"], prefix="contract")
        tenure_dummies = pd.get_dummies(df["tenure_category"], prefix="tenuregroup")
        age_dummies = pd.get_dummies(df["age_group"], prefix="agegroup")
        spend_dummies = pd.get_dummies(df["spend_category"], prefix="spendcategory")

        df = pd.concat(
            [
                df,
                subscription_dummies,
                contract_dummies,
                tenure_dummies,
                age_dummies,
                spend_dummies,
            ],
            axis=1,
        )

        # Drop original categorical columns and ID
        columns_to_drop = [
            "subscription_type",
            "contract_length",
            "tenure_category",
            "age_group",
            "spend_category",
        ]

        # Also drop customerid if present
        if "customerid" in df.columns:
            columns_to_drop.append("customerid")

        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        df = df.drop(columns=columns_to_drop)

        return df






# Ensures your input has EXACT same columns as training data (adds missing with zeros)
    def align_features(
        self, df: pd.DataFrame, expected_features: List[str]
    ) -> pd.DataFrame:

        df = df.copy()

        # Add missing features with zeros
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0

        # Keep only expected features in the correct order
        df = df[expected_features]

        return df





# Main pipeline that runs all steps in order like clean , process, feature engineering, encoding and alignment
    def preprocess(
        self, data: Any, expected_features: Optional[List[str]] = None
    ) -> pd.DataFrame:

        # Convert input to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Step 1: Clean column names
        df = self.clean_column_names(df)

        # Step 2: Handle missing values and basic preprocessing
        df = raw_preprocess(df, strategy="auto")

        # Step 3: Build engineered features
        df = build_feature(df)

        # Step 4: Encode categorical variables
        df = self.encode_features(df)

        # Step 5: Align with expected features if provided
        if expected_features:
            df = self.align_features(df, expected_features)

        return df










# ============================================================================
# MODEL LOADER class : Heandles loading ML models from various sourcs.
# ============================================================================


class ModelLoader:
    """Heandles loading ML models from various sourcs."""



#  Load specific model by run ID
    @staticmethod
    def load_from_mlflow_run(
        run_id: str, tracking_uri: str = "./mlruns"
    ) -> Tuple[Any, List[str]]:
       
        mlflow.set_tracking_uri(tracking_uri)

        try:
            model_uri = f"runs:/{run_id}/model"
            logger.info(f"Loading model from MLflow run: {run_id}")
            model = mlflow.xgboost.load_model(model_uri)

            # Extract feature names
            feature_names = ModelLoader._extract_feature_names(model)

            logger.info(f"Model loaded successfully. Features: {len(feature_names)}")
            return model, feature_names

        except Exception as e:
            logger.error(f"Failed to load model from MLflow run {run_id}: {e}")
            raise




# Automatically finds best model by recall metric
    @staticmethod
    def load_best_from_experiment(
        experiment_name: str, tracking_uri: str = "./mlruns", metric: str = "recall"
    ) -> Tuple[Any, List[str], str]:

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        # Get experiment
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        # Search for best run
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1,
        )

        if not runs:
            raise ValueError(f"No runs found in experiment '{experiment_name}'")

        best_run = runs[0]
        run_id = best_run.info.run_id
        metric_value = best_run.data.metrics.get(metric, 0)

        logger.info(f"Loading best model from experiment '{experiment_name}'")
        logger.info(f"Run ID: {run_id}, {metric}={metric_value:.4f}")

        # Load the model
        model, feature_names = ModelLoader.load_from_mlflow_run(run_id, tracking_uri)

        return model, feature_names, run_id





# Load from .pkl or .joblib file 
    @staticmethod
    def load_from_file(model_path: str) -> Tuple[Any, List[str]]:

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            logger.info(f"Loading model from file: {model_path}")
            model = joblib.load(model_path)

            # Extract feature names
            feature_names = ModelLoader._extract_feature_names(model)

            logger.info("Model loaded successfully from file")
            return model, feature_names

        except Exception as e:
            logger.error(f"Failed to load model from file: {e}")
            raise





# Extract feature names from a model.
    @staticmethod
    def _extract_feature_names(model: Any) -> List[str]:

        feature_names = []

        try:
            # Try different attributes
            if hasattr(model, "feature_names_in_"):
                feature_names = list(model.feature_names_in_)
            elif hasattr(model, "feature_names"):
                feature_names = list(model.feature_names)
            elif hasattr(model, "get_booster"):
                booster = model.get_booster()
                if hasattr(booster, "feature_names"):
                    feature_names = list(booster.feature_names)
        except Exception as e:
            logger.warning(f"Could not extract feature names: {e}")

        return feature_names







# ============================================================================
# INFERENCE ENGINE :  Main class that brings everything together
# ============================================================================


class ChurnPredictor:

    def __init__(
        self,
        model: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
        threshold: float = 0.5,
    ):

        self.model = model
        self.feature_names = feature_names or []
        self.threshold = threshold
        self.preprocessor = InferencePreprocessor()
        self.model_info = {}



# bring model from  MLflow using run ID
    def load_model_from_mlflow_run(
        self, run_id: str, tracking_uri: str = "./mlruns"
    ) -> None:

        self.model, self.feature_names = ModelLoader.load_from_mlflow_run(
            run_id, tracking_uri
        )
        self.model_info = {
            "source": "mlflow_run",
            "run_id": run_id,
            "tracking_uri": tracking_uri,
        }


# bring best model from experiment using experiment name and metric to optimize
    def load_best_model_from_experiment(
        self,
        experiment_name: str,
        tracking_uri: str = "./mlruns",
        metric: str = "recall",
    ) -> None:
        """

        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking URI
            metric: Metric to optimize
        """
        self.model, self.feature_names, run_id = ModelLoader.load_best_from_experiment(
            experiment_name, tracking_uri, metric
        )
        self.model_info = {
            "source": "mlflow_experiment",
            "experiment_name": experiment_name,
            "run_id": run_id,
            "tracking_uri": tracking_uri,
            "optimized_metric": metric,
        }


# Load model from file (e.g. .pkl or .joblib)
    def load_model_from_file(self, model_path: str) -> None:
        """
        Load model from file.

        Args:
            model_path: Path to model file
        """
        self.model, self.feature_names = ModelLoader.load_from_file(model_path)
        self.model_info = {"source": "file", "path": model_path}



# Predict churn for a single customer
    def predict(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model_* method first.")

        # Preprocess
        X = self.preprocessor.preprocess(customer_data, self.feature_names)

        # Predict probability
        proba = self.model.predict_proba(X)[:, 1][0]

        # Binary prediction
        prediction = int(proba >= self.threshold)

        # Risk level
        risk_level = self._calculate_risk_level(proba)

        return {
            "customerid": customer_data.get("customerid")
            or customer_data.get("CustomerID"),
            "churn_probability": round(float(proba), 4),
            "churn_prediction": prediction,
            "risk_level": risk_level,
        }




# Predict churn for a batch of customers
    def predict_batch(self, customers: List[Dict[str, Any]]) -> Dict[str, Any]:

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model_* method first.")

        # Convert to DataFrame
        df = pd.DataFrame(customers)
        customer_ids = df.get("customerid", df.get("CustomerID", [None] * len(df)))

        # Preprocess
        X = self.preprocessor.preprocess(df, self.feature_names)

        # Predict probabilities
        probas = self.model.predict_proba(X)[:, 1]
        predictions = (probas >= self.threshold).astype(int)

        # Build results
        results = []
        high_risk_count = 0

        for i, (proba, pred) in enumerate(zip(probas, predictions)):
            risk_level = self._calculate_risk_level(proba)

            if risk_level in ["High", "Critical"]:
                high_risk_count += 1

            results.append(
                {
                    "customerid": customer_ids.iloc[i]
                    if isinstance(customer_ids, pd.Series)
                    else customer_ids[i],
                    "churn_probability": round(float(proba), 4),
                    "churn_prediction": int(pred),
                    "risk_level": risk_level,
                }
            )

        return {
            "predictions": results,
            "total_customers": len(customers),
            "high_risk_count": high_risk_count,
            "churn_rate": round(sum(predictions) / len(predictions), 4),
        }


# Calculate risk level based on churn probability
    def _calculate_risk_level(self, probability: float) -> str:

        if probability >= 0.8:
            return "Critical"
        elif probability >= 0.6:
            return "High"
        elif probability >= 0.4:
            return "Medium"
        else:
            return "Low"


# model meta data information for health checks and monitoring
    def get_model_info(self) -> Dict[str, Any]:

        return {
            **self.model_info,
            "feature_count": len(self.feature_names),
            "threshold": self.threshold,
            "model_type": type(self.model).__name__ if self.model else None,
            "model_loaded": self.model is not None,
        }








# ============================================================================
# CONVENIENCE FUNCTIONS : this call the main class and provide easy ways to create predictor from different sources
# ============================================================================


def create_predictor_from_mlflow(
    run_id: Optional[str] = None,
    experiment_name: Optional[str] = None,
    tracking_uri: str = "./mlruns",
    metric: str = "recall",
    threshold: float = 0.5,
) -> ChurnPredictor:
    """
    Create and initialize a ChurnPredictor from MLflow.

    Args:
        run_id: Specific run ID (optional)
        experiment_name: Experiment name to load best model (optional)
        tracking_uri: MLflow tracking URI
        metric: Metric to optimize when loading from experiment
        threshold: Prediction threshold

    Returns:
        Initialized ChurnPredictor
    """
    predictor = ChurnPredictor(threshold=threshold)

    if run_id:
        predictor.load_model_from_mlflow_run(run_id, tracking_uri)
    elif experiment_name:
        predictor.load_best_model_from_experiment(experiment_name, tracking_uri, metric)
    else:
        raise ValueError("Either run_id or experiment_name must be provided")

    return predictor


def create_predictor_from_file(
    model_path: str, threshold: float = 0.5
) -> ChurnPredictor:
    """
    Create and initialize a ChurnPredictor from a file.

    Args:
        model_path: Path to model file
        threshold: Prediction threshold

    Returns:
        Initialized ChurnPredictor
    """
    predictor = ChurnPredictor(threshold=threshold)
    predictor.load_model_from_file(model_path)
    return predictor









# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Example: Load best model from experiment
    predictor = create_predictor_from_mlflow(
        experiment_name="Colab_GPU_Training",
        tracking_uri="./mlruns",
        metric="recall",
        threshold=0.5,
    )

    # Example customer data
    customer = {
        "customerid": "CUST001",
        "age": 35,
        "gender": "Male",
        "tenure": 24,
        "usage_frequency": 15,
        "support_calls": 3,
        "payment_delay": 5,
        "subscription_type": "Premium",
        "contract_length": "Annual",
        "total_spend": 1250.50,
        "last_interaction": 10,
    }

    # Make prediction
    result = predictor.predict(customer)
    print("\nPrediction Result:")
    print(result)

    # Get model info
    print("\nModel Info:")
    print(predictor.get_model_info())
