"""
Test suite for feature engineering modules
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build_feature import build_feature
from src.features.feature_preprocess import preprocess_features


class TestBuildFeature:
    """Test feature engineering functionality"""

    @pytest.fixture
    def sample_data(self):
        """Create sample preprocessed data"""
        return pd.DataFrame(
            {
                "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
                "gender": [
                    "Male",
                    "Female",
                    "Male",
                    "Female",
                    "Male",
                    "Female",
                    "Male",
                    "Female",
                    "Male",
                    "Female",
                ],
                "tenure": [6, 12, 18, 24, 30, 36, 42, 48, 54, 60],
                "usage_frequency": [10, 15, 20, 25, 8, 30, 5, 35, 40, 2],
                "support_calls": [2, 3, 1, 4, 5, 2, 6, 1, 3, 7],
                "payment_delay": [0, 5, 2, 10, 15, 3, 20, 0, 8, 25],
                "subscription_type": [
                    "Basic",
                    "Premium",
                    "Standard",
                    "Premium",
                    "Basic",
                    "Standard",
                    "Premium",
                    "Basic",
                    "Standard",
                    "Premium",
                ],
                "contract_length": [
                    "Monthly",
                    "Annual",
                    "Quarterly",
                    "Annual",
                    "Monthly",
                    "Quarterly",
                    "Annual",
                    "Monthly",
                    "Quarterly",
                    "Annual",
                ],
                "total_spend": [
                    500,
                    1500,
                    1000,
                    2000,
                    300,
                    1200,
                    2500,
                    600,
                    1800,
                    3000,
                ],
                "last_interaction": [5, 10, 3, 15, 20, 8, 25, 2, 12, 30],
                "churn": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

    def test_build_feature_returns_dataframe(self, sample_data):
        """Test that build_feature returns a DataFrame"""
        result = build_feature(sample_data)
        assert isinstance(result, pd.DataFrame)

    def test_build_feature_preserves_rows(self, sample_data):
        """Test that build_feature preserves number of rows"""
        result = build_feature(sample_data)
        assert len(result) == len(sample_data)

    def test_clv_feature_created(self, sample_data):
        """Test that CLV feature is created"""
        result = build_feature(sample_data)
        assert "CLV" in result.columns or "clv" in result.columns.str.lower()

    def test_clv_calculation(self, sample_data):
        """Test CLV calculation is correct"""
        result = build_feature(sample_data)
        clv_col = [col for col in result.columns if "clv" in col.lower()][0]

        # CLV should be total_spend / tenure
        expected_clv = (
            sample_data["total_spend"].iloc[0] / sample_data["tenure"].iloc[0]
        )
        assert abs(result[clv_col].iloc[0] - expected_clv) < 0.01

    def test_support_efficiency_created(self, sample_data):
        """Test that support_efficiency feature is created"""
        result = build_feature(sample_data)
        assert any("support_efficiency" in col.lower() for col in result.columns)

    def test_payment_reliability_created(self, sample_data):
        """Test that payment_reliability feature is created"""
        result = build_feature(sample_data)
        assert any("payment_reliability" in col.lower() for col in result.columns)

    def test_payment_reliability_range(self, sample_data):
        """Test payment_reliability is in valid range (0-1)"""
        result = build_feature(sample_data)
        reliability_col = [
            col for col in result.columns if "payment_reliability" in col.lower()
        ][0]

        assert result[reliability_col].min() >= 0
        assert result[reliability_col].max() <= 1

    def test_engagement_score_created(self, sample_data):
        """Test that engagement_score feature is created"""
        result = build_feature(sample_data)
        assert any("engagement_score" in col.lower() for col in result.columns)

    def test_value_to_company_created(self, sample_data):
        """Test that value_to_company feature is created"""
        result = build_feature(sample_data)
        assert any("value_to_company" in col.lower() for col in result.columns)

    def test_tenure_category_created(self, sample_data):
        """Test that tenure_category feature is created"""
        result = build_feature(sample_data)
        assert any(
            "tenure" in col.lower() and "category" in col.lower()
            for col in result.columns
        ) or any("tenuregroup" in col.lower() for col in result.columns)

    def test_age_group_created(self, sample_data):
        """Test that age_group feature is created"""
        result = build_feature(sample_data)
        assert any(
            "age" in col.lower() and "group" in col.lower() for col in result.columns
        ) or any("agegroup" in col.lower() for col in result.columns)

    def test_spend_category_created(self, sample_data):
        """Test that spend_category feature is created"""
        result = build_feature(sample_data)
        assert any(
            "spend" in col.lower() and "category" in col.lower()
            for col in result.columns
        ) or any("spendcategory" in col.lower() for col in result.columns)

    def test_no_infinite_values(self, sample_data):
        """Test that no infinite values are created"""
        result = build_feature(sample_data)
        numeric_cols = result.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            assert not np.isinf(result[col]).any(), (
                f"Column {col} contains infinite values"
            )

    def test_no_nan_in_engineered_features(self, sample_data):
        """Test that engineered features don't introduce NaN"""
        result = build_feature(sample_data)

        # Original data has no NaN
        if sample_data.isnull().sum().sum() == 0:
            # New features should also have no NaN
            new_cols = set(result.columns) - set(sample_data.columns)
            for col in new_cols:
                assert not result[col].isnull().any(), (
                    f"Engineered feature {col} contains NaN"
                )


class TestPreprocessFeatures:
    """Test feature preprocessing functionality"""

    @pytest.fixture
    def sample_featured_data(self):
        """Create sample data with engineered features"""
        df = pd.DataFrame(
            {
                "age": [25, 30, 35, 40, 45],
                "gender": ["Male", "Female", "Male", "Female", "Male"],
                "tenure": [12, 24, 36, 48, 60],
                "usage_frequency": [10, 15, 20, 25, 30],
                "support_calls": [2, 3, 1, 4, 5],
                "payment_delay": [0, 5, 2, 10, 15],
                "subscription_type": [
                    "Basic",
                    "Premium",
                    "Standard",
                    "Premium",
                    "Basic",
                ],
                "contract_length": [
                    "Monthly",
                    "Annual",
                    "Quarterly",
                    "Annual",
                    "Monthly",
                ],
                "total_spend": [500, 1500, 1000, 2000, 300],
                "last_interaction": [5, 10, 3, 15, 20],
                "CLV": [41.67, 62.5, 27.78, 41.67, 5.0],
                "support_efficiency": [0.2, 0.2, 0.05, 0.16, 0.17],
                "payment_reliability": [1.0, 0.17, 0.33, 0.09, 0.06],
                "engagement_score": [1.67, 1.36, 5.0, 1.56, 1.43],
                "value_to_company": [166.67, 375.0, 500.0, 400.0, 50.0],
                "Tenure Category": "12-24 Months",
                "Age Group": "18-30",
                "Spend Category": "Medium",
                "churn": [0, 1, 0, 1, 0],
            }
        )
        return build_feature(df)

    def test_preprocess_features_returns_tuple(self, sample_featured_data):
        """Test that preprocess_features returns tuple of (X, y, feature_names)"""
        result = preprocess_features(sample_featured_data)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_preprocess_features_X_shape(self, sample_featured_data):
        """Test that X has correct shape"""
        X, y, feature_names = preprocess_features(sample_featured_data)

        assert len(X) == len(sample_featured_data)
        assert X.shape[1] > 0

    def test_preprocess_features_y_shape(self, sample_featured_data):
        """Test that y has correct shape"""
        X, y, feature_names = preprocess_features(sample_featured_data)

        assert len(y) == len(sample_featured_data)

    def test_preprocess_features_names_match_X(self, sample_featured_data):
        """Test that feature_names length matches X columns"""
        X, y, feature_names = preprocess_features(sample_featured_data)

        assert len(feature_names) == X.shape[1]

    def test_gender_encoded(self, sample_featured_data):
        """Test that gender is encoded"""
        X, y, feature_names = preprocess_features(sample_featured_data)

        # Gender should be encoded as numeric
        if "gender" in feature_names:
            gender_idx = feature_names.index("gender")
            assert X[:, gender_idx].dtype in [
                np.int64,
                np.int32,
                np.float64,
                np.float32,
            ]

    def test_categorical_one_hot_encoded(self, sample_featured_data):
        """Test that categorical features are one-hot encoded"""
        X, y, feature_names = preprocess_features(sample_featured_data)

        # Check for encoded subscription type columns
        assert any(
            "sub_" in name.lower() or "subscription" in name.lower()
            for name in feature_names
        )

        # Check for encoded contract length columns
        assert any("contract_" in name.lower() for name in feature_names)

    def test_no_categorical_columns_in_output(self, sample_featured_data):
        """Test that original categorical columns are removed"""
        X, y, feature_names = preprocess_features(sample_featured_data)

        # These should not be in feature names (they should be encoded)
        assert not any(name.lower() == "subscription_type" for name in feature_names)
        assert not any(name.lower() == "contract_length" for name in feature_names)

    def test_all_numeric_features(self, sample_featured_data):
        """Test that all features in X are numeric"""
        X, y, feature_names = preprocess_features(sample_featured_data)

        assert X.dtype in [np.int64, np.int32, np.float64, np.float32]

    def test_no_missing_values_in_X(self, sample_featured_data):
        """Test that X has no missing values"""
        X, y, feature_names = preprocess_features(sample_featured_data)

        assert not np.isnan(X).any(), "X should not contain NaN values"

    def test_no_missing_values_in_y(self, sample_featured_data):
        """Test that y has no missing values"""
        X, y, feature_names = preprocess_features(sample_featured_data)

        assert not np.isnan(y).any(), "y should not contain NaN values"


class TestFeatureEngineeringIntegration:
    """Integration tests for complete feature engineering pipeline"""

    @pytest.fixture
    def sample_clean_data(self):
        """Create clean preprocessed data"""
        return pd.DataFrame(
            {
                "age": [25, 30, 35, 40, 45, 50],
                "gender": ["Male", "Female", "Male", "Female", "Male", "Female"],
                "tenure": [12, 24, 36, 48, 60, 6],
                "usage_frequency": [10, 15, 20, 25, 30, 5],
                "support_calls": [2, 3, 1, 4, 5, 6],
                "payment_delay": [0, 5, 2, 10, 15, 20],
                "subscription_type": [
                    "Basic",
                    "Premium",
                    "Standard",
                    "Premium",
                    "Basic",
                    "Standard",
                ],
                "contract_length": [
                    "Monthly",
                    "Annual",
                    "Quarterly",
                    "Annual",
                    "Monthly",
                    "Quarterly",
                ],
                "total_spend": [500, 1500, 1000, 2000, 300, 800],
                "last_interaction": [5, 10, 3, 15, 20, 25],
                "churn": [0, 1, 0, 1, 0, 1],
            }
        )

    def test_full_pipeline(self, sample_clean_data):
        """Test complete feature engineering pipeline"""
        # Build features
        df_featured = build_feature(sample_clean_data)
        assert len(df_featured) == len(sample_clean_data)

        # Preprocess features
        X, y, feature_names = preprocess_features(df_featured)

        assert len(X) == len(sample_clean_data)
        assert len(y) == len(sample_clean_data)
        assert len(feature_names) == X.shape[1]

    def test_pipeline_produces_ml_ready_data(self, sample_clean_data):
        """Test that pipeline produces ML-ready data"""
        df_featured = build_feature(sample_clean_data)
        X, y, feature_names = preprocess_features(df_featured)

        # All numeric
        assert X.dtype in [np.int64, np.int32, np.float64, np.float32]

        # No missing values
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()

        # Target is binary
        assert set(np.unique(y)) <= {0, 1}

    def test_pipeline_reproducibility(self, sample_clean_data):
        """Test that pipeline is reproducible"""
        # Run 1
        df_featured1 = build_feature(sample_clean_data.copy())
        X1, y1, features1 = preprocess_features(df_featured1)

        # Run 2
        df_featured2 = build_feature(sample_clean_data.copy())
        X2, y2, features2 = preprocess_features(df_featured2)

        # Should produce identical results
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
        assert features1 == features2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
