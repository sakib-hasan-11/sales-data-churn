"""
Create a test model for CI/CD testing
"""

import joblib
import numpy as np
from sklearn.datasets import make_classification
from xgboost import XGBClassifier


def create_test_model():
    """Create a simple test model and save it"""

    # Create synthetic data
    X, y = make_classification(
        n_samples=100, n_features=20, n_informative=15, n_redundant=5, random_state=42
    )

    # Create feature names (simulating actual features)
    feature_names = [
        "age",
        "gender",
        "tenure",
        "usage_frequency",
        "support_calls",
        "payment_delay",
        "total_spend",
        "last_interaction",
        "CLV",
        "support_efficiency",
        "payment_reliability",
        "engagement_score",
        "value_to_company",
        "sub_Basic",
        "sub_Premium",
        "sub_Standard",
        "contract_Monthly",
        "contract_Annual",
        "contract_Quarterly",
        "tenuregroup_12_24M",
    ]

    # Train a simple model
    model = XGBClassifier(
        n_estimators=10,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )

    model.fit(X, y)

    # Save model and features
    model_data = {"model": model, "feature_names": feature_names}

    joblib.dump(model_data, "tests/test_model.pkl")
    print("✅ Test model created successfully: tests/test_model.pkl")

    return model, feature_names


if __name__ == "__main__":
    create_test_model()
