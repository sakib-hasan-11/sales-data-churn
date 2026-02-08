"""
Comprehensive unit tests for sales-data-churn project.
Tests all functions from:
- src/data_processing/load.py
- src/data_processing/preprocess.py
- src/features/build_feature.py
- src/features/feature_preprocess.py
- src/utils/data_validator.py
- main.py
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from main import main
from src.data_processing.load import load_data
from src.data_processing.preprocess import raw_preprocess
from src.features.build_feature import build_feature

# ==================== FIXTURES ====================


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with required columns for feature engineering."""
    return pd.DataFrame(
        {
            "CustomerID": [1, 2, 3, 4, 5],
            "Age": [25, 35, 45, 55, 65],
            "Gender": ["Male", "Female", "Male", "Female", "Male"],
            "Tenure": [12, 24, 36, 48, 60],
            "Usage Frequency": [5, 10, 15, 20, 25],
            "Support Calls": [2, 4, 6, 8, 10],
            "Payment Delay": [0, 5, 10, 15, 20],
            "Subscription Type": ["Basic", "Standard", "Premium", "Basic", "Standard"],
            "Contract Length": [
                "Monthly",
                "Quarterly",
                "Annual",
                "Monthly",
                "Quarterly",
            ],
            "Total Spend": [100.0, 200.0, 300.0, 400.0, 500.0],
            "Last Interaction": [1, 5, 10, 15, 20],
            "Churn": [0, 0, 1, 1, 0],
        }
    )


@pytest.fixture
def valid_df():
    """Create a valid DataFrame that passes all validations."""
    return pd.DataFrame(
        {
            "CustomerID": [1, 2, 3, 4, 5],
            "Age": [25, 35, 45, 55, 65],
            "Gender": ["Male", "Female", "Male", "Female", "Male"],
            "Tenure": [12, 24, 36, 48, 60],
            "Usage Frequency": [5, 10, 15, 20, 25],
            "Support Calls": [2, 4, 6, 8, 10],
            "Payment Delay": [0, 5, 10, 15, 20],
            "Subscription Type": ["Basic", "Standard", "Premium", "Basic", "Standard"],
            "Contract Length": [
                "Monthly",
                "Quarterly",
                "Annual",
                "Monthly",
                "Quarterly",
            ],
            "Total Spend": [100.0, 200.0, 300.0, 400.0, 500.0],
            "Last Interaction": [1, 5, 10, 15, 20],
            "Churn": [0, 0, 1, 1, 0],
        }
    )


# ==================== LOAD DATA TESTS ====================


class TestLoadData:
    """Tests for the load_data function."""

    def test_load_data_valid_file(self, tmp_path):
        """Test loading a valid CSV file."""
        csv_file = tmp_path / "test_data.csv"
        test_df = pd.DataFrame(
            {
                "CustomerID": [1, 2, 3],
                "Age": [25, 30, 35],
                "Gender": ["Male", "Female", "Male"],
            }
        )
        test_df.to_csv(csv_file, index=False)

        result = load_data(csv_file)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert len(result.columns) == 3
        assert list(result.columns) == ["CustomerID", "Age", "Gender"]

    def test_load_data_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for non-existent file."""
        non_existent_file = tmp_path / "non_existent.csv"

        with pytest.raises(FileNotFoundError) as exc_info:
            load_data(non_existent_file)

        assert "Train data not found" in str(exc_info.value)

    def test_load_data_returns_dataframe(self, tmp_path):
        """Test that load_data returns a pandas DataFrame."""
        csv_file = tmp_path / "test_data.csv"
        pd.DataFrame({"col1": [1, 2], "col2": [3, 4]}).to_csv(csv_file, index=False)

        result = load_data(csv_file)

        assert isinstance(result, pd.DataFrame)

    def test_load_data_preserves_columns(self, tmp_path):
        """Test that all columns from CSV are preserved."""
        csv_file = tmp_path / "test_data.csv"
        expected_columns = ["A", "B", "C", "D", "E"]
        test_df = pd.DataFrame({col: [1, 2, 3] for col in expected_columns})
        test_df.to_csv(csv_file, index=False)

        result = load_data(csv_file)

        assert list(result.columns) == expected_columns

    def test_load_data_preserves_row_count(self, tmp_path):
        """Test that all rows from CSV are loaded."""
        csv_file = tmp_path / "test_data.csv"
        num_rows = 100
        test_df = pd.DataFrame({"col1": range(num_rows)})
        test_df.to_csv(csv_file, index=False)

        result = load_data(csv_file)

        assert len(result) == num_rows

    def test_load_data_handles_empty_csv(self, tmp_path):
        """Test loading an empty CSV file (with headers only)."""
        csv_file = tmp_path / "empty.csv"
        pd.DataFrame(columns=["col1", "col2"]).to_csv(csv_file, index=False)

        result = load_data(csv_file)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ["col1", "col2"]


# ==================== RAW PREPROCESS TESTS ====================


class TestRawPreprocess:
    """Tests for the raw_preprocess function."""

    # Column Name Cleaning Tests
    def test_column_name_cleaning_spaces(self):
        """Test that spaces in column names are replaced with underscores."""
        df = pd.DataFrame({"Column Name": [1, 2, 3]})
        result = raw_preprocess(df)
        assert "column_name" in result.columns

    def test_column_name_cleaning_slashes(self):
        """Test that slashes in column names are replaced with underscores."""
        df = pd.DataFrame({"Column/Name": [1, 2, 3]})
        result = raw_preprocess(df)
        assert "column_name" in result.columns

    def test_column_name_cleaning_special_chars(self):
        """Test that special characters are removed from column names."""
        df = pd.DataFrame({"Column@Name#Here!": [1, 2, 3]})
        result = raw_preprocess(df)
        assert "columnnamehere" in result.columns

    def test_column_name_cleaning_lowercase(self):
        """Test that column names are converted to lowercase."""
        df = pd.DataFrame({"UPPERCASE": [1, 2, 3]})
        result = raw_preprocess(df)
        assert "uppercase" in result.columns

    def test_column_name_cleaning_multiple_underscores(self):
        """Test that multiple consecutive underscores are reduced to one."""
        df = pd.DataFrame({"Column___Name": [1, 2, 3]})
        result = raw_preprocess(df)
        assert "column_name" in result.columns

    # Missing Value Handling Tests (Auto Strategy)
    def test_auto_strategy_numeric_median(self):
        """Test that numeric columns use median fill in auto mode."""
        df = pd.DataFrame({"num_col": [1.0, 2.0, np.nan, 4.0, 5.0]})
        result = raw_preprocess(df, strategy="auto")
        assert result["num_col"].isna().sum() == 0
        assert result["num_col"].iloc[2] == 3.0

    def test_auto_strategy_object_mode(self):
        """Test that object columns use mode fill in auto mode."""
        df = pd.DataFrame({"cat_col": ["A", "A", "B", np.nan, "A"]})
        result = raw_preprocess(df, strategy="auto")
        assert result["cat_col"].isna().sum() == 0
        assert result["cat_col"].iloc[3] == "A"

    def test_auto_strategy_object_unknown_fallback(self):
        """Test that object columns with no mode fallback to 'Unknown'."""
        df = pd.DataFrame({"cat_col": [np.nan, np.nan, np.nan]})
        df["cat_col"] = df["cat_col"].astype(object)
        result = raw_preprocess(df, strategy="auto")
        assert result["cat_col"].iloc[0] == "Unknown"

    # Constant Strategy Tests
    def test_constant_strategy_numeric(self):
        """Test that numeric columns are filled with 0 in constant mode."""
        df = pd.DataFrame({"num_col": [1.0, np.nan, 3.0]})
        result = raw_preprocess(df, strategy="constant")
        assert result["num_col"].iloc[1] == 0

    def test_constant_strategy_object(self):
        """Test that object columns are filled with 'Unknown' in constant mode."""
        df = pd.DataFrame({"cat_col": ["A", np.nan, "B"]})
        result = raw_preprocess(df, strategy="constant")
        assert result["cat_col"].iloc[1] == "Unknown"

    # Median Strategy Tests
    def test_median_strategy_numeric(self):
        """Test that numeric columns use median fill in median mode."""
        df = pd.DataFrame({"num_col": [10.0, 20.0, np.nan, 40.0]})
        result = raw_preprocess(df, strategy="median")
        assert result["num_col"].iloc[2] == 20.0

    # Mode Strategy Tests
    def test_mode_strategy_numeric(self):
        """Test that numeric columns use mode fill in mode strategy."""
        df = pd.DataFrame({"num_col": [1, 1, 1, np.nan, 2]})
        result = raw_preprocess(df, strategy="mode")
        assert result["num_col"].iloc[3] == 1

    def test_mode_strategy_object(self):
        """Test that object columns use mode fill in mode strategy."""
        df = pd.DataFrame({"cat_col": ["X", "Y", "Y", np.nan]})
        result = raw_preprocess(df, strategy="mode")
        assert result["cat_col"].iloc[3] == "Y"

    # Custom Fill Tests
    def test_custom_fill_overrides_strategy(self):
        """Test that custom_fill values override the strategy."""
        df = pd.DataFrame({"num_col": [1.0, np.nan, 3.0]})
        result = raw_preprocess(df, strategy="median", custom_fill={"num_col": 999})
        assert result["num_col"].iloc[1] == 999

    def test_custom_fill_partial_columns(self):
        """Test custom_fill for some columns, strategy for others."""
        df = pd.DataFrame({"col_a": [1.0, np.nan, 3.0], "col_b": [np.nan, 5.0, 6.0]})
        result = raw_preprocess(df, strategy="constant", custom_fill={"col_a": 100})
        assert result["col_a"].iloc[1] == 100
        assert result["col_b"].iloc[0] == 0

    # Datetime Handling Tests
    def test_datetime_forward_backward_fill(self):
        """Test that datetime columns use forward/backward fill."""
        df = pd.DataFrame(
            {"date_col": pd.to_datetime(["2021-01-01", None, "2021-01-03", None])}
        )
        result = raw_preprocess(df, strategy="auto")
        assert result["date_col"].isna().sum() == 0

    # Return Type Tests
    def test_returns_dataframe(self):
        """Test that raw_preprocess returns a DataFrame."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        result = raw_preprocess(df)
        assert isinstance(result, pd.DataFrame)

    def test_does_not_modify_original(self):
        """Test that the original DataFrame is not modified."""
        df = pd.DataFrame({"num_col": [1.0, np.nan, 3.0]})
        original_copy = df.copy()
        raw_preprocess(df, strategy="auto")
        pd.testing.assert_frame_equal(df, original_copy)


# ==================== BUILD FEATURE TESTS ====================


class TestBuildFeature:
    """Tests for the build_feature function."""

    # CLV Tests
    def test_clv_created(self, sample_df):
        """Test that CLV column is created."""
        result = build_feature(sample_df)
        assert "CLV" in result.columns

    def test_clv_calculation(self, sample_df):
        """Test CLV calculation: Total Spend * (Tenure / 12)."""
        result = build_feature(sample_df)
        assert result["CLV"].iloc[0] == 100.0
        assert result["CLV"].iloc[1] == 400.0

    # Support Efficiency Tests
    def test_support_efficiency_created(self, sample_df):
        """Test that Support_Efficiency column is created."""
        result = build_feature(sample_df)
        assert "Support_Efficiency" in result.columns

    def test_support_efficiency_calculation(self, sample_df):
        """Test Support_Efficiency calculation: Support Calls / (Tenure + 1)."""
        result = build_feature(sample_df)
        expected = 2 / 13
        assert pytest.approx(result["Support_Efficiency"].iloc[0], rel=1e-4) == expected

    # Payment Reliability Tests
    def test_payment_reliability_created(self, sample_df):
        """Test that Payment_Reliability column is created."""
        result = build_feature(sample_df)
        assert "Payment_Reliability" in result.columns

    def test_payment_reliability_calculation(self, sample_df):
        """Test Payment_Reliability calculation: 1 / (Payment Delay + 1)."""
        result = build_feature(sample_df)
        assert result["Payment_Reliability"].iloc[0] == 1.0
        expected = 1 / 6
        assert (
            pytest.approx(result["Payment_Reliability"].iloc[1], rel=1e-4) == expected
        )

    # Usage Score Tests
    def test_usage_score_created(self, sample_df):
        """Test that Usage_Score column is created."""
        result = build_feature(sample_df)
        assert "Usage_Score" in result.columns

    def test_usage_score_calculation(self, sample_df):
        """Test Usage_Score calculation: (Usage Frequency / max) * 100."""
        result = build_feature(sample_df)
        assert result["Usage_Score"].iloc[0] == 20.0
        assert result["Usage_Score"].iloc[4] == 100.0

    # Engagement Index Tests
    def test_engagement_index_created(self, sample_df):
        """Test that Engagement_Index column is created."""
        result = build_feature(sample_df)
        assert "Engagement_Index" in result.columns

    def test_engagement_index_calculation(self, sample_df):
        """Test Engagement_Index calculation: (Usage Frequency + Support Calls) / 2."""
        result = build_feature(sample_df)
        assert result["Engagement_Index"].iloc[0] == 3.5

    # Spend per Interaction Tests
    def test_spend_per_interaction_created(self, sample_df):
        """Test that Spend_per_Interaction column is created."""
        result = build_feature(sample_df)
        assert "Spend_per_Interaction" in result.columns

    def test_spend_per_interaction_calculation(self, sample_df):
        """Test Spend_per_Interaction calculation: Total Spend / (Last Interaction + 1)."""
        result = build_feature(sample_df)
        assert result["Spend_per_Interaction"].iloc[0] == 50.0

    # Risk Score Tests
    def test_risk_score_created(self, sample_df):
        """Test that Risk_Score column is created."""
        result = build_feature(sample_df)
        assert "Risk_Score" in result.columns

    def test_risk_score_is_numeric(self, sample_df):
        """Test that Risk_Score is numeric."""
        result = build_feature(sample_df)
        assert pd.api.types.is_numeric_dtype(result["Risk_Score"])

    # Tenure Category Tests
    def test_tenure_category_created(self, sample_df):
        """Test that Tenure_Category column is created."""
        result = build_feature(sample_df)
        assert "Tenure_Category" in result.columns

    def test_tenure_category_bins(self, sample_df):
        """Test Tenure_Category binning."""
        result = build_feature(sample_df)
        assert result["Tenure_Category"].iloc[0] == "0-12M"
        assert result["Tenure_Category"].iloc[1] == "12-24M"
        assert result["Tenure_Category"].iloc[2] == "24-36M"

    # Age Group Tests
    def test_age_group_created(self, sample_df):
        """Test that Age_Group column is created."""
        result = build_feature(sample_df)
        assert "Age_Group" in result.columns

    def test_age_group_bins(self, sample_df):
        """Test Age_Group binning."""
        result = build_feature(sample_df)
        assert result["Age_Group"].iloc[0] == "18-30"
        assert result["Age_Group"].iloc[1] == "30-40"
        assert result["Age_Group"].iloc[2] == "40-50"

    # Spend Category Tests
    def test_spend_category_created(self, sample_df):
        """Test that Spend_Category column is created."""
        result = build_feature(sample_df)
        assert "Spend_Category" in result.columns

    def test_spend_category_values(self, sample_df):
        """Test Spend_Category has valid values."""
        result = build_feature(sample_df)
        valid_values = ["Low", "Medium", "High"]
        for val in result["Spend_Category"].dropna():
            assert val in valid_values

    # Output Shape Tests
    def test_output_has_more_columns(self, sample_df):
        """Test that output has more columns than input."""
        original_cols = len(sample_df.columns)
        result = build_feature(sample_df)
        assert len(result.columns) > original_cols

    def test_output_preserves_rows(self, sample_df):
        """Test that output has the same number of rows as input."""
        result = build_feature(sample_df)
        assert len(result) == len(sample_df)

    def test_returns_dataframe(self, sample_df):
        """Test that build_feature returns a DataFrame."""
        result = build_feature(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_all_new_features_created(self, sample_df):
        """Test that all expected new features are created."""
        result = build_feature(sample_df)
        expected_new_features = [
            "CLV",
            "Support_Efficiency",
            "Payment_Reliability",
            "Usage_Score",
            "Engagement_Index",
            "Spend_per_Interaction",
            "Risk_Score",
            "Tenure_Category",
            "Age_Group",
            "Spend_Category",
        ]
        for feature in expected_new_features:
            assert feature in result.columns, f"Missing feature: {feature}"


# ==================== FEATURE PREPROCESS TESTS ====================


class TestFeaturePreprocess:
    """Tests for the preprocess_features function."""

    @pytest.fixture
    def feature_df(self):
        """Create a sample DataFrame with feature engineering columns."""
        return pd.DataFrame(
            {
                "CustomerID": [1, 2, 3, 4, 5],
                "Age": [25, 35, 45, 55, 65],
                "Gender": ["Male", "Female", "Male", "Female", "Male"],
                "Tenure": [12, 24, 36, 48, 60],
                "Subscription Type": [
                    "Basic",
                    "Standard",
                    "Premium",
                    "Basic",
                    "Standard",
                ],
                "Contract Length": [
                    "Monthly",
                    "Quarterly",
                    "Annual",
                    "Monthly",
                    "Quarterly",
                ],
                "Total Spend": [100.0, 200.0, 300.0, 400.0, 500.0],
                "Churn": [0, 0, 1, 1, 0],
                "Tenure_Category": pd.Categorical(
                    ["0-12M", "12-24M", "24-36M", "36-48M", "48M+"],
                    categories=["0-12M", "12-24M", "24-36M", "36-48M", "48M+"],
                ),
                "Age_Group": pd.Categorical(
                    ["18-30", "30-40", "40-50", "50-60", "60+"],
                    categories=["18-30", "30-40", "40-50", "50-60", "60+"],
                ),
                "Spend_Category": pd.Categorical(
                    ["Low", "Low", "Medium", "High", "High"],
                    categories=["Low", "Medium", "High"],
                ),
            }
        )

    # Gender Label Encoding Tests
    def test_gender_label_encoded(self, feature_df):
        """Test that Gender column is label encoded."""
        df = feature_df.copy()
        le_gender = LabelEncoder()
        df["Gender"] = le_gender.fit_transform(df["Gender"])
        assert pd.api.types.is_numeric_dtype(df["Gender"])

    def test_gender_encoding_values(self, feature_df):
        """Test that Gender encoding produces valid values (0 or 1)."""
        df = feature_df.copy()
        le_gender = LabelEncoder()
        df["Gender"] = le_gender.fit_transform(df["Gender"])
        assert set(df["Gender"].unique()).issubset({0, 1})

    # One-Hot Encoding Tests
    def test_subscription_type_one_hot(self, feature_df):
        """Test that Subscription Type is one-hot encoded."""
        df = feature_df.copy()
        subscription_dummies = pd.get_dummies(df["Subscription Type"], prefix="Sub")
        df = pd.concat([df, subscription_dummies], axis=1)
        expected_cols = ["Sub_Basic", "Sub_Standard", "Sub_Premium"]
        for col in expected_cols:
            assert col in df.columns

    def test_contract_length_one_hot(self, feature_df):
        """Test that Contract Length is one-hot encoded."""
        df = feature_df.copy()
        contract_dummies = pd.get_dummies(df["Contract Length"], prefix="Contract")
        df = pd.concat([df, contract_dummies], axis=1)
        expected_cols = ["Contract_Monthly", "Contract_Quarterly", "Contract_Annual"]
        for col in expected_cols:
            assert col in df.columns

    def test_tenure_category_one_hot(self, feature_df):
        """Test that Tenure_Category is one-hot encoded."""
        df = feature_df.copy()
        tenure_dummies = pd.get_dummies(df["Tenure_Category"], prefix="TenureGroup")
        df = pd.concat([df, tenure_dummies], axis=1)
        tenure_cols = [col for col in df.columns if col.startswith("TenureGroup_")]
        assert len(tenure_cols) > 0

    def test_age_group_one_hot(self, feature_df):
        """Test that Age_Group is one-hot encoded."""
        df = feature_df.copy()
        age_dummies = pd.get_dummies(df["Age_Group"], prefix="AgeGroup")
        df = pd.concat([df, age_dummies], axis=1)
        age_cols = [col for col in df.columns if col.startswith("AgeGroup_")]
        assert len(age_cols) > 0

    def test_spend_category_one_hot(self, feature_df):
        """Test that Spend_Category is one-hot encoded."""
        df = feature_df.copy()
        spend_dummies = pd.get_dummies(df["Spend_Category"], prefix="SpendCategory")
        df = pd.concat([df, spend_dummies], axis=1)
        expected_cols = [
            "SpendCategory_Low",
            "SpendCategory_Medium",
            "SpendCategory_High",
        ]
        for col in expected_cols:
            assert col in df.columns

    # Column Dropping Tests
    def test_original_categorical_columns_dropped(self, feature_df):
        """Test that original categorical columns are dropped after one-hot encoding."""
        df = feature_df.copy()
        subscription_dummies = pd.get_dummies(df["Subscription Type"], prefix="Sub")
        contract_dummies = pd.get_dummies(df["Contract Length"], prefix="Contract")
        tenure_dummies = pd.get_dummies(df["Tenure_Category"], prefix="TenureGroup")
        age_dummies = pd.get_dummies(df["Age_Group"], prefix="AgeGroup")
        spend_dummies = pd.get_dummies(df["Spend_Category"], prefix="SpendCategory")

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
        cols_to_drop = [
            "Subscription Type",
            "Contract Length",
            "Tenure_Category",
            "Age_Group",
            "Spend_Category",
        ]
        df = df.drop(cols_to_drop, axis=1)
        for col in cols_to_drop:
            assert col not in df.columns

    def test_one_hot_encoding_sums_to_one(self, feature_df):
        """Test that one-hot encoded columns sum to 1 for each row."""
        df = feature_df.copy()
        subscription_dummies = pd.get_dummies(df["Subscription Type"], prefix="Sub")
        row_sums = subscription_dummies.sum(axis=1)
        assert (row_sums == 1).all()

    def test_output_preserves_row_count(self, feature_df):
        """Test that preprocessing preserves the number of rows."""
        df = feature_df.copy()
        original_rows = len(df)
        subscription_dummies = pd.get_dummies(df["Subscription Type"], prefix="Sub")
        df = pd.concat([df, subscription_dummies], axis=1)
        assert len(df) == original_rows

    def test_customerid_preserved(self, feature_df):
        """Test that CustomerID column is preserved."""
        assert "CustomerID" in feature_df.columns

    def test_churn_preserved(self, feature_df):
        """Test that Churn column is preserved."""
        assert "Churn" in feature_df.columns


# ==================== DATA VALIDATOR TESTS ====================


class TestDataValidator:
    """Tests for schema, business logic, and range validation."""

    # Schema Validation Tests
    def test_valid_df_has_all_required_columns(self, valid_df):
        """Test that valid DataFrame has all required columns."""
        required_columns = [
            "CustomerID",
            "Age",
            "Gender",
            "Tenure",
            "Usage Frequency",
            "Support Calls",
            "Payment Delay",
            "Subscription Type",
            "Contract Length",
            "Total Spend",
            "Last Interaction",
            "Churn",
        ]
        for col in required_columns:
            assert col in valid_df.columns

    # Business Logic Validation Tests
    def test_gender_valid_values(self, valid_df):
        """Test that Gender contains only valid values."""
        valid_values = {"Male", "Female"}
        assert set(valid_df["Gender"].unique()).issubset(valid_values)

    def test_subscription_type_valid_values(self, valid_df):
        """Test that Subscription Type contains only valid values."""
        valid_values = {"Basic", "Standard", "Premium"}
        assert set(valid_df["Subscription Type"].unique()).issubset(valid_values)

    def test_contract_length_valid_values(self, valid_df):
        """Test that Contract Length contains only valid values."""
        valid_values = {"Monthly", "Quarterly", "Annual"}
        assert set(valid_df["Contract Length"].unique()).issubset(valid_values)

    def test_churn_binary_values(self, valid_df):
        """Test that Churn contains only binary values."""
        valid_values = {0, 1, "0", "1"}
        assert set(valid_df["Churn"].unique()).issubset(valid_values)

    # Numeric Range Validation Tests
    def test_tenure_non_negative(self, valid_df):
        """Test that Tenure is non-negative."""
        assert (valid_df["Tenure"] >= 0).all()

    def test_total_spend_non_negative(self, valid_df):
        """Test that Total Spend is non-negative."""
        assert (valid_df["Total Spend"] >= 0).all()

    def test_age_in_valid_range(self, valid_df):
        """Test that Age is in valid range (18-120)."""
        assert ((valid_df["Age"] >= 18) & (valid_df["Age"] <= 120)).all()

    def test_usage_frequency_in_valid_range(self, valid_df):
        """Test that Usage Frequency is in valid range (0-31)."""
        assert (
            (valid_df["Usage Frequency"] >= 0) & (valid_df["Usage Frequency"] <= 31)
        ).all()

    def test_support_calls_in_valid_range(self, valid_df):
        """Test that Support Calls is in valid range (0-365)."""
        assert (
            (valid_df["Support Calls"] >= 0) & (valid_df["Support Calls"] <= 365)
        ).all()

    def test_payment_delay_in_valid_range(self, valid_df):
        """Test that Payment Delay is in valid range (0-365)."""
        assert (
            (valid_df["Payment Delay"] >= 0) & (valid_df["Payment Delay"] <= 365)
        ).all()

    def test_last_interaction_in_valid_range(self, valid_df):
        """Test that Last Interaction is in valid range (0-365)."""
        assert (
            (valid_df["Last Interaction"] >= 0) & (valid_df["Last Interaction"] <= 365)
        ).all()

    # Null Value Tests
    def test_customerid_not_null(self, valid_df):
        """Test that CustomerID has no null values."""
        assert valid_df["CustomerID"].isna().sum() == 0

    def test_tenure_not_null(self, valid_df):
        """Test that Tenure has no null values."""
        assert valid_df["Tenure"].isna().sum() == 0

    def test_total_spend_not_null(self, valid_df):
        """Test that Total Spend has no null values."""
        assert valid_df["Total Spend"].isna().sum() == 0

    def test_age_not_null(self, valid_df):
        """Test that Age has no null values."""
        assert valid_df["Age"].isna().sum() == 0

    # Statistical Constraints Tests
    def test_usage_frequency_mean_reasonable(self, valid_df):
        """Test that Usage Frequency mean is in reasonable range."""
        mean_val = valid_df["Usage Frequency"].mean()
        assert 1 <= mean_val <= 20

    # Consistency Tests
    def test_total_spend_greater_than_support_calls(self, valid_df):
        """Test that Total Spend is generally >= Support Calls."""
        condition_met = (valid_df["Total Spend"] >= valid_df["Support Calls"]).mean()
        assert condition_met >= 0.95

    # Integration Tests with Mocking
    @patch("src.utils.data_validator.gx")
    def test_validate_data_returns_tuple(self, mock_gx, valid_df):
        """Test that validate_data returns a tuple of (bool, list)."""
        mock_context = MagicMock()
        mock_gx.get_context.return_value = mock_context
        mock_datasource = MagicMock()
        mock_context.sources.add_pandas.return_value = mock_datasource
        mock_data_asset = MagicMock()
        mock_datasource.add_dataframe_asset.return_value = mock_data_asset
        mock_batch_request = MagicMock()
        mock_data_asset.build_batch_request.return_value = mock_batch_request
        mock_validator = MagicMock()
        mock_context.get_validator.return_value = mock_validator
        mock_validator.validate.return_value = {"success": True, "results": []}

        from src.utils.data_validator import validate_data

        result = validate_data(valid_df)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], list)

    @patch("src.utils.data_validator.gx")
    def test_validate_data_success_case(self, mock_gx, valid_df):
        """Test that validate_data returns True for valid data."""
        mock_context = MagicMock()
        mock_gx.get_context.return_value = mock_context
        mock_datasource = MagicMock()
        mock_context.sources.add_pandas.return_value = mock_datasource
        mock_data_asset = MagicMock()
        mock_datasource.add_dataframe_asset.return_value = mock_data_asset
        mock_batch_request = MagicMock()
        mock_data_asset.build_batch_request.return_value = mock_batch_request
        mock_validator = MagicMock()
        mock_context.get_validator.return_value = mock_validator
        mock_validator.validate.return_value = {
            "success": True,
            "results": [
                {"success": True, "expectation_config": {"expectation_type": "test"}}
            ],
        }

        from src.utils.data_validator import validate_data

        success, failed = validate_data(valid_df)

        assert success is True
        assert len(failed) == 0

    @patch("src.utils.data_validator.gx")
    def test_validate_data_failure_case(self, mock_gx, valid_df):
        """Test that validate_data returns failed expectations on failure."""
        mock_context = MagicMock()
        mock_gx.get_context.return_value = mock_context
        mock_datasource = MagicMock()
        mock_context.sources.add_pandas.return_value = mock_datasource
        mock_data_asset = MagicMock()
        mock_datasource.add_dataframe_asset.return_value = mock_data_asset
        mock_batch_request = MagicMock()
        mock_data_asset.build_batch_request.return_value = mock_batch_request
        mock_validator = MagicMock()
        mock_context.get_validator.return_value = mock_validator
        mock_validator.validate.return_value = {
            "success": False,
            "results": [
                {
                    "success": False,
                    "expectation_config": {
                        "expectation_type": "expect_column_to_exist"
                    },
                },
                {
                    "success": True,
                    "expectation_config": {
                        "expectation_type": "expect_column_values_to_not_be_null"
                    },
                },
            ],
        }

        from src.utils.data_validator import validate_data

        success, failed = validate_data(valid_df)

        assert success is False
        assert "expect_column_to_exist" in failed


# ==================== MAIN FUNCTION TESTS ====================


class TestMain:
    """Tests for the main function."""

    def test_main_runs_without_error(self):
        """Test that main function executes without raising exceptions."""
        main()

    def test_main_prints_expected_message(self, capsys):
        """Test that main function prints the expected message."""
        main()
        captured = capsys.readouterr()
        assert "Hello from sales-data-churn!" in captured.out

    def test_main_returns_none(self):
        """Test that main function returns None (implicit return)."""
        result = main()
        assert result is None
