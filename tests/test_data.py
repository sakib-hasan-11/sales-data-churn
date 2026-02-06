from pathlib import Path

import pandas as pd
import pytest

from src.data_processing.load import load_data
from src.data_processing.preprocess import raw_preprocess
from src.features.build_feature import build_feature
from src.features.feature_preprocess import preprocess_features
from src.training.mlflow_training import train_xgboost_with_mlflow
from src.training.optuna_tuning import optimize_xgboost_hyperparameters

# =====================================================
# 1️ LOAD RAW DATA
# =====================================================


@pytest.fixture(scope="session")
def raw_data():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "tests" / "dummy_data.csv"

    df = load_data(data_path)
    return df


def test_raw_data_loaded(raw_data):
    assert raw_data is not None
    assert raw_data.shape[0] > 0
    assert "Churn" in raw_data.columns


# =====================================================
# 2️ RAW PREPROCESS
# =====================================================


@pytest.fixture(scope="session")
def processed_data(raw_data):
    df_processed = raw_preprocess(raw_data)
    return df_processed


def test_no_null_after_raw_preprocess(processed_data):
    assert processed_data.isnull().sum().sum() == 0


# =====================================================
# 3️ BUILD FEATURES
# =====================================================


@pytest.fixture(scope="session")
def feature_data(processed_data):
    df_features = build_feature(processed_data)
    return df_features


EXPECTED_FEATURES = [
    "clv",
    "support_efficiency",
    "payment_reliability",
    "usage_score",
    "engagement_index",
    "spend_per_interaction",
    "risk_score",
    "tenure_category",
    "age_group",
    "spend_category",
]


def test_feature_columns_present(feature_data):
    for col in EXPECTED_FEATURES:
        assert col in feature_data.columns


def test_no_null_in_features_except_categories(feature_data):
    allowed_nulls = {
        "tenure_category",
        "age_group",
        "spend_category",
    }

    cols_to_check = [c for c in feature_data.columns if c not in allowed_nulls]

    assert feature_data[cols_to_check].isnull().sum().sum() == 0


# =====================================================
# 4️ FEATURE PREPROCESS (ONE-HOT + DROP + SAVE)
# =====================================================


@pytest.fixture(scope="module")
def preprocessed_feature_data(feature_data, tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("preprocessed")

    df = preprocess_features(feature_data, output_dir, "preprocessed_features.csv")

    return {
        "df": df,
        "output_dir": output_dir,
        "output_file": output_dir / "preprocessed_features.csv",
    }


def test_preprocessed_file_created(preprocessed_feature_data):
    assert preprocessed_feature_data["output_file"].exists()


def test_no_null_after_feature_preprocess(preprocessed_feature_data):
    df = preprocessed_feature_data["df"]
    assert df.isnull().sum().sum() == 0


# =====================================================
# 5️ ONE-HOT ENCODING VALIDATION
# =====================================================

DROPPED_COLUMNS = [
    "subscription_type",
    "contract_length",
    "tenure_category",
    "age_group",
    "spend_category",
]


def test_target_column_present(preprocessed_feature_data):
    df = preprocessed_feature_data["df"]
    assert "churn" in df.columns


def test_original_categorical_columns_removed(preprocessed_feature_data):
    df = preprocessed_feature_data["df"]
    for col in DROPPED_COLUMNS:
        assert col not in df.columns


EXPECTED_PREFIXES = [
    "sub_",
    "contract_",
    "tenuregroup_",
    "agegroup_",
    "spendcategory_",
]


def test_dummy_columns_created(preprocessed_feature_data):
    df = preprocessed_feature_data["df"]
    columns = df.columns.tolist()

    for prefix in EXPECTED_PREFIXES:
        assert any(col.startswith(prefix) for col in columns), (
            f"No column found with prefix {prefix}"
        )


def test_one_hot_values_are_binary(preprocessed_feature_data):
    df = preprocessed_feature_data["df"]

    dummy_columns = [
        col
        for col in df.columns
        if col.startswith(
            ("sub_", "contract_", "tenuregroup_", "agegroup_", "spendcategory_")
        )
    ]

    for col in dummy_columns:
        unique_vals = set(df[col].unique())
        assert unique_vals.issubset({0, 1}), f"{col} has non-binary values"


def test_one_hot_sum_rule(preprocessed_feature_data):
    df = preprocessed_feature_data["df"]

    sub_cols = [c for c in df.columns if c.startswith("Sub_")]
    assert (df[sub_cols].sum(axis=1) <= 1).all()


# =====================================================
# TRAIN TEST SPLIT (REUSABLE FIXTURE) # no need for productin as we will use different test and train data .
# =====================================================
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="module")
def train_test_split_data(preprocessed_feature_data, target_col):
    """
    Create train-test split once and reuse across:
    - optuna
    - mlflow training
    Avoid redundant splitting.
    """
    df = preprocessed_feature_data["df"]

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[target_col],
    )

    return train_df, test_df


@pytest.fixture(scope="module")
def train_data(train_test_split_data):
    return train_test_split_data[0]


@pytest.fixture(scope="module")
def test_data(train_test_split_data):
    return train_test_split_data[1]


# =====================================================
# 6️ OPTUNA HYPERPARAMETER TUNING
# =====================================================

import pytest
import os
import pickle
import mlflow

from src.training.mlflow_training import train_xgboost_with_mlflow
from src.training.optuna_tuning import optimize_xgboost_hyperparameters


@pytest.fixture(scope="session")
def target_col():
    return "churn"


# -----------------------------------------------------
# Optuna tuning fixture (runs once)
# -----------------------------------------------------
@pytest.fixture(scope="module")
def run_optuna_tuning(
    train_data,
    test_data,
    target_col,
    n_trials=10,
    optimize_metric="recall",
):
    return optimize_xgboost_hyperparameters(
        train_data=train_data,
        test_data=test_data,
        target_col=target_col,
        n_trials=n_trials,
        optimize_metric=optimize_metric,
    )


def test_optuna_tuning(run_optuna_tuning):
    """
    Validate optuna tuning output only.
    """
    result = run_optuna_tuning

    assert result is not None
    assert isinstance(result["best_params"], dict)
    assert isinstance(result["best_score"], float)
    assert isinstance(result["all_metrics"], dict)


# =====================================================
# TEMP ARTIFACT DIR (mlruns + model save)
# Auto deleted after test
# =====================================================
@pytest.fixture(scope="function")
def temp_artifact_dir(tmp_path):
    """
    Creates temp directory for:
    - mlruns
    - model save
    Auto removed after test.
    """

    base_dir = tmp_path

    mlruns_dir = base_dir / "mlruns"
    model_dir = base_dir / "models"

    mlruns_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    return {
        "mlruns": str(mlruns_dir),
        "model_dir": str(model_dir),
    }


# =====================================================
# MLflow + Training Integration Test
# =====================================================
def test_train_with_mlflow_using_optuna_output(
    train_data,
    test_data,
    target_col,
    run_optuna_tuning,
    temp_artifact_dir,
):
    """
    Full pipeline test:
    optuna → training → mlflow logging → model save
    Works local + CI + container.
    """

    assert run_optuna_tuning is not None
    assert "best_params" in run_optuna_tuning

    mlruns_path = temp_artifact_dir["mlruns"]
    model_dir = temp_artifact_dir["model_dir"]

    # ---------------------------
    # Run training
    # ---------------------------
    results = train_xgboost_with_mlflow(
        train_data=train_data,
        test_data=test_data,
        target_col=target_col,
        threshold_value=0.5,
        experiment_name="CI_XGB_MLFLOW_TEST",
        n_optuna_trials=1,   # small for CI/local
        n_runs=1,
        mlflow_tracking_uri=f"file:{mlruns_path}",
        model_save_dir=model_dir,   # VERY IMPORTANT CHANGE
    )

    # ---------------------------
    # Basic result check
    # ---------------------------
    assert results is not None
    assert isinstance(results, dict)

    # ---------------------------
    # Check model saved
    # ---------------------------
    safe_threshold = str(0.5).replace(".", "_")
    model_filename = f"model_threshold_{safe_threshold}.pkl"

    model_path = os.path.join(model_dir, model_filename)

    assert os.path.exists(model_path), "Model not saved"


    with open(model_path, "rb") as f:
        model = pickle.load(f)

    assert model is not None

    # ---------------------------
    # Check mlruns created
    # ---------------------------
    assert os.path.exists(mlruns_path)
    assert len(os.listdir(mlruns_path)) > 0

    # ---------------------------
    # Check MLflow experiment
    # ---------------------------
    mlflow.set_tracking_uri(f"file:{mlruns_path}")

    experiment = mlflow.get_experiment_by_name("CI_XGB_MLFLOW_TEST")
    assert experiment is not None

    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment.experiment_id)

    assert len(runs) > 0

    metrics = runs[0].data.metrics

    expected_metrics = ["recall", "precision", "f1_score", "accuracy", "auc"]

    for metric in expected_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], float)
