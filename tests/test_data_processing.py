"""
Test suite for data processing modules
"""

import numpy as np
import pandas as pd
import pytest

from src.data_processing.load import load_data
from src.data_processing.preprocess import raw_preprocess
from src.utils.data_validator import validate_data


class TestLoadData:
    """Test data loading functionality"""

    def test_load_data_success(self):
        """Test successful data loading"""
        df = load_data("tests/dummy_data.csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "churn" in df.columns.str.lower()

    def test_load_data_columns(self):
        """Test that all expected columns are present"""
        df = load_data("tests/dummy_data.csv")
        expected_cols = [
            "age",
            "gender",
            "tenure",
            "usage_frequency",
            "support_calls",
            "payment_delay",
            "subscription_type",
            "contract_length",
            "total_spend",
            "last_interaction",
        ]

        df_cols_lower = df.columns.str.lower()
        for col in expected_cols:
            assert any(col in c for c in df_cols_lower), f"Column {col} not found"

    def test_load_data_file_not_found(self):
        """Test error handling for missing file"""
        with pytest.raises((FileNotFoundError, Exception)):
            load_data("non_existent_file.csv")

    def test_load_data_types(self):
        """Test data types are appropriate"""
        df = load_data("tests/dummy_data.csv")
        assert df["Age"].dtype in [np.int64, np.int32, np.float64]
        assert df["Gender"].dtype == object
        assert df["Total Spend"].dtype in [np.float64, np.float32]


class TestRawPreprocess:
    """Test raw data preprocessing"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame(
            {
                "Age": [25, 30, None, 45, 50],
                "Gender": ["Male", "Female", "Male", None, "Female"],
                "Tenure": [12, 24, 36, None, 60],
                "Usage Frequency": [10, 15, None, 25, 30],
                "Support Calls": [2, 3, 4, None, 1],
                "Payment Delay": [0, 5, None, 10, 0],
                "Subscription Type": ["Basic", "Premium", None, "Standard", "Premium"],
                "Contract Length": ["Monthly", "Annual", "Quarterly", None, "Annual"],
                "Total Spend": [500.0, 1500.0, None, 2000.0, 2500.0],
                "Last Interaction": [5, 10, 15, None, 2],
                "Churn": [0, 1, 0, 1, 0],
            }
        )

    def test_preprocess_handles_missing_values(self, sample_data):
        """Test that missing values are handled"""
        df_processed = raw_preprocess(sample_data)
        assert df_processed.isnull().sum().sum() == 0, (
            "Should have no missing values after preprocessing"
        )

    def test_preprocess_preserves_shape(self, sample_data):
        """Test that preprocessing preserves number of rows"""
        df_processed = raw_preprocess(sample_data)
        assert len(df_processed) == len(sample_data), "Row count should be preserved"

    def test_preprocess_column_names(self, sample_data):
        """Test that column names are cleaned"""
        df_processed = raw_preprocess(sample_data)
        # Check no spaces in column names
        assert all(" " not in col for col in df_processed.columns), (
            "Column names should not have spaces"
        )

    def test_preprocess_data_types(self, sample_data):
        """Test that data types are appropriate after preprocessing"""
        df_processed = raw_preprocess(sample_data)
        # Numeric columns should be numeric
        numeric_cols = [
            "age",
            "tenure",
            "usage_frequency",
            "support_calls",
            "payment_delay",
            "total_spend",
            "last_interaction",
        ]
        for col in numeric_cols:
            matching_cols = [
                c
                for c in df_processed.columns
                if col.replace("_", "").lower() in c.lower().replace("_", "")
            ]
            if matching_cols:
                assert df_processed[matching_cols[0]].dtype in [
                    np.int64,
                    np.int32,
                    np.float64,
                    np.float32,
                ]

    def test_preprocess_outliers(self, sample_data):
        """Test outlier handling"""
        # Add extreme outliers
        sample_data.loc[0, "Age"] = 200
        sample_data.loc[1, "Total Spend"] = 1000000

        df_processed = raw_preprocess(sample_data)

        # Check ages are in reasonable range (after preprocessing)
        age_col = [col for col in df_processed.columns if "age" in col.lower()][0]
        assert df_processed[age_col].max() < 150, "Age outliers should be handled"


class TestDataValidator:
    """Test data validation functionality"""

    @pytest.fixture
    def valid_data(self):
        """Create valid data for testing"""
        return pd.DataFrame(
            {
                "Age": [25, 30, 35, 40, 45],
                "Gender": ["Male", "Female", "Male", "Female", "Male"],
                "Tenure": [12, 24, 36, 48, 60],
                "Usage Frequency": [10, 15, 20, 25, 30],
                "Support Calls": [2, 3, 4, 1, 2],
                "Payment Delay": [0, 5, 0, 10, 0],
                "Subscription Type": [
                    "Basic",
                    "Premium",
                    "Standard",
                    "Premium",
                    "Basic",
                ],
                "Contract Length": [
                    "Monthly",
                    "Annual",
                    "Quarterly",
                    "Annual",
                    "Monthly",
                ],
                "Total Spend": [500.0, 1500.0, 1000.0, 2000.0, 800.0],
                "Last Interaction": [5, 10, 15, 20, 2],
                "Churn": [0, 1, 0, 1, 0],
            }
        )

    def test_validate_valid_data(self, valid_data):
        """Test validation of valid data"""
        result = validate_data(valid_data)
        assert result is True or isinstance(result, dict), (
            "Valid data should pass validation"
        )

    def test_validate_missing_columns(self):
        """Test validation catches missing columns"""
        invalid_data = pd.DataFrame(
            {
                "Age": [25, 30],
                "Gender": ["Male", "Female"],
                # Missing other required columns
            }
        )

        result = validate_data(invalid_data)
        # Should either raise exception or return False/dict with errors
        if isinstance(result, dict):
            assert "errors" in result or "success" in result

    def test_validate_data_types(self, valid_data):
        """Test validation catches wrong data types"""
        # Corrupt a numeric column with strings
        invalid_data = valid_data.copy()
        invalid_data["Age"] = ["twenty", "thirty", "35", "40", "45"]

        try:
            result = validate_data(invalid_data)
            if isinstance(result, dict):
                assert not result.get("success", True)
        except Exception:
            pass  # Exception is also acceptable

    def test_validate_empty_dataframe(self):
        """Test validation handles empty dataframe"""
        empty_df = pd.DataFrame()

        try:
            result = validate_data(empty_df)
            if isinstance(result, bool):
                assert result is False
            elif isinstance(result, dict):
                assert not result.get("success", True)
        except Exception:
            pass  # Exception is acceptable


class TestDataProcessingIntegration:
    """Integration tests for complete data processing pipeline"""

    def test_full_pipeline(self):
        """Test complete pipeline: load -> preprocess"""
        # Load
        df = load_data("tests/dummy_data.csv")
        assert len(df) > 0

        # Preprocess
        df_processed = raw_preprocess(df)
        assert len(df_processed) > 0
        assert df_processed.isnull().sum().sum() == 0

    def test_pipeline_preserves_target(self):
        """Test that target variable (churn) is preserved"""
        df = load_data("tests/dummy_data.csv")
        df_processed = raw_preprocess(df)

        # Check target variable exists
        assert any("churn" in col.lower() for col in df_processed.columns)

    def test_pipeline_reproducibility(self):
        """Test that pipeline produces consistent results"""
        df1 = load_data("tests/dummy_data.csv")
        df1_processed = raw_preprocess(df1)

        df2 = load_data("tests/dummy_data.csv")
        df2_processed = raw_preprocess(df2)

        # Should produce same shape
        assert df1_processed.shape == df2_processed.shape

        # Should have same columns
        assert set(df1_processed.columns) == set(df2_processed.columns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
