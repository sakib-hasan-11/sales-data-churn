"""XGBoost hyperparameter optimization using Optuna."""

from typing import Any, Dict

import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def optimize_xgboost_hyperparameters(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    target_col: str,
    n_trials: int = 100,
    optimize_metric: str = "recall",
) -> Dict[str, Any]:
    """
    Optimize XGBoost hyperparameters using Optuna.

    Parameters
    ----------
    train_data : pd.DataFrame
        Training dataset with features and target column
    test_data : pd.DataFrame
        Test dataset with features and target column
    target_col : str
        Name of the target column
    n_trials : int, optional
        Number of optimization trials (default: 100)
    optimize_metric : str, optional
        Metric to optimize: 'recall', 'precision', 'f1', 'auc' (default: 'recall')

    Returns
    -------
    Dict[str, Any]
        Dictionary containing best hyperparameters and performance metrics
    """

    # Separate features and target
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]

    def objective(trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization."""

        # Define hyperparameter search space
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
            "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "random_state": 42,
            "tree_method": "hist",
            "device": "cuda",
            "verbosity": 0,
        }

        # Add dart-specific parameters
        if params["booster"] == "dart":
            params["rate_drop"] = trial.suggest_float("rate_drop", 0.0, 0.5)
            params["skip_drop"] = trial.suggest_float("skip_drop", 0.0, 0.5)

        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Predict probabilities
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        # Calculate metrics
        if optimize_metric == "recall":
            score = recall_score(y_test, y_pred, zero_division=0)
        elif optimize_metric == "precision":
            score = precision_score(y_test, y_pred, zero_division=0)
        elif optimize_metric == "f1":
            score = f1_score(y_test, y_pred, zero_division=0)
        elif optimize_metric == "auc":
            score = roc_auc_score(y_test, y_pred_proba)
        else:
            raise ValueError(f"Unknown metric: {optimize_metric}")

        return score

    # Create study and optimize
    study = optuna.create_study(
        direction="maximize", study_name=f"xgboost_optimization_{optimize_metric}"
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Get best parameters
    best_params = study.best_params
    best_score = study.best_value

    # Train final model with best parameters
    final_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "tree_method": "hist",
        "device": "cuda",
        "verbosity": 0,
        **best_params,
    }

    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X_train, y_train)

    # Calculate all metrics on test set
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]
    y_pred = final_model.predict(X_test)

    metrics = {
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_pred_proba),
    }

    # Return results
    results = {
        "best_params": best_params,
        "best_score": best_score,
        "optimized_metric": optimize_metric,
        "all_metrics": metrics,
        "n_trials": n_trials,
        "study": study,
        "final_model": final_model,
    }

    return results
