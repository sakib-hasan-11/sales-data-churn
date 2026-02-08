"""
Test suite for inference engine
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from src.inference.inference import (
    ChurnPredictor,
    InferencePreprocessor,
    ModelLoader,
    create_predictor_from_file,
)


class TestInferencePreprocessor:
    """Test InferencePreprocessor class"""

    @pytest.fixture
    def preprocessor(self):
        return InferencePreprocessor()

    @pytest.fixture
    def sample_customer(self):
        return {
            "customerid": "TEST001",
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

    def test_clean_column_names(self, preprocessor):
        """Test column name cleaning"""
        df = pd.DataFrame(
            {
                "Customer ID": [1, 2],
                "Total Spend": [100, 200],
                "Usage/Frequency": [5, 10],
            }
        )

        cleaned = preprocessor.clean_column_names(df)

        assert "customerid" in cleaned.columns or "customer_id" in cleaned.columns
        assert all(" " not in col and "/" not in col for col in cleaned.columns)

    def test_encode_features(self, preprocessor, sample_customer):
        """Test feature encoding"""
        df = pd.DataFrame([sample_customer])
        df = preprocessor.clean_column_names(df)

        # Mock the necessary preprocessing
        df["Tenure Category"] = "12-24 Months"
        df["Age Group"] = "30-40"
        df["Spend Category"] = "High"

        encoded = preprocessor.encode_features(df)

        # Gender should be encoded to numeric
        assert encoded["gender"].dtype in [np.int64, np.int32, np.float64]

        # One-hot encoded columns should exist
        assert any("sub_" in col.lower() for col in encoded.columns)

    def test_align_features(self, preprocessor):
        """Test feature alignment"""
        df = pd.DataFrame({"age": [35], "tenure": [24], "gender": [0]})

        expected_features = [
            "age",
            "tenure",
            "gender",
            "new_feature",
            "another_feature",
        ]

        aligned = preprocessor.align_features(df, expected_features)

        assert list(aligned.columns) == expected_features
        assert aligned["new_feature"].iloc[0] == 0
        assert aligned["another_feature"].iloc[0] == 0

    def test_preprocess_dict_input(self, preprocessor, sample_customer):
        """Test preprocessing with dict input"""
        expected_features = ["age", "tenure", "gender", "usage_frequency"]

        result = preprocessor.preprocess(sample_customer, expected_features)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_preprocess_dataframe_input(self, preprocessor, sample_customer):
        """Test preprocessing with DataFrame input"""
        df = pd.DataFrame([sample_customer])
        expected_features = ["age", "tenure", "gender", "usage_frequency"]

        result = preprocessor.preprocess(df, expected_features)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1


class TestModelLoader:
    """Test ModelLoader class"""

    def test_load_from_file(self):
        """Test loading model from file"""
        model, features = ModelLoader.load_from_file("tests/test_model.pkl")

        assert model is not None
        assert isinstance(features, list)
        assert len(features) > 0

    def test_load_from_file_invalid_path(self):
        """Test error handling for invalid file path"""
        with pytest.raises(FileNotFoundError):
            ModelLoader.load_from_file("non_existent_model.pkl")


class TestChurnPredictor:
    """Test ChurnPredictor class"""

    @pytest.fixture
    def predictor(self):
        """Create predictor with test model"""
        predictor = ChurnPredictor()
        predictor.load_model_from_file("tests/test_model.pkl")
        return predictor

    @pytest.fixture
    def sample_customer(self):
        return {
            "customerid": "TEST001",
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

    def test_load_model_from_file(self):
        """Test model loading from file"""
        predictor = ChurnPredictor()
        predictor.load_model_from_file("tests/test_model.pkl")

        assert predictor.model is not None
        assert predictor.feature_names is not None
        assert len(predictor.feature_names) > 0

    def test_predict_single(self, predictor, sample_customer):
        """Test single customer prediction"""
        result = predictor.predict(sample_customer)

        assert "customerid" in result
        assert "churn_probability" in result
        assert "churn_prediction" in result
        assert "risk_level" in result

        assert 0 <= result["churn_probability"] <= 1
        assert result["churn_prediction"] in [0, 1]
        assert result["risk_level"] in ["Low", "Medium", "High", "Critical"]

    def test_predict_batch(self, predictor, sample_customer):
        """Test batch prediction"""
        customers = [
            sample_customer,
            {**sample_customer, "customerid": "TEST002", "age": 45},
            {**sample_customer, "customerid": "TEST003", "age": 25},
        ]

        result = predictor.predict_batch(customers)

        assert "predictions" in result
        assert "total_customers" in result
        assert "high_risk_count" in result
        assert "churn_rate" in result

        assert len(result["predictions"]) == 3
        assert result["total_customers"] == 3

    def test_calculate_risk_level(self, predictor):
        """Test risk level calculation"""
        assert predictor._calculate_risk_level(0.9) == "Critical"
        assert predictor._calculate_risk_level(0.7) == "High"
        assert predictor._calculate_risk_level(0.5) == "Medium"
        assert predictor._calculate_risk_level(0.2) == "Low"

    def test_get_model_info(self, predictor):
        """Test model info retrieval"""
        info = predictor.get_model_info()

        assert isinstance(info, dict)
        assert "model_loaded" in info
        assert info["model_loaded"] is True


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_create_predictor_from_file(self):
        """Test creating predictor from file"""
        predictor = create_predictor_from_file("tests/test_model.pkl")

        assert predictor.model is not None
        assert predictor.feature_names is not None


class TestInferenceIntegration:
    """Integration tests for complete inference pipeline"""

    @pytest.fixture
    def sample_customers(self):
        return [
            {
                "customerid": f"TEST{i:03d}",
                "age": 30 + i * 5,
                "gender": "Male" if i % 2 == 0 else "Female",
                "tenure": 12 + i * 6,
                "usage_frequency": 10 + i * 2,
                "support_calls": 2 + i % 5,
                "payment_delay": i % 10,
                "subscription_type": ["Basic", "Premium", "Standard"][i % 3],
                "contract_length": ["Monthly", "Annual", "Quarterly"][i % 3],
                "total_spend": 500 + i * 200,
                "last_interaction": 5 + i * 2,
            }
            for i in range(5)
        ]

    def test_end_to_end_prediction(self, sample_customers):
        """Test complete prediction pipeline"""
        predictor = create_predictor_from_file("tests/test_model.pkl")

        # Single prediction
        result = predictor.predict(sample_customers[0])
        assert "churn_probability" in result

        # Batch prediction
        batch_result = predictor.predict_batch(sample_customers)
        assert len(batch_result["predictions"]) == len(sample_customers)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
