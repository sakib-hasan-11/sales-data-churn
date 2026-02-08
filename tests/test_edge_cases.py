"""
Test suite for edge cases and error handling
"""

import numpy as np
import pandas as pd
import pytest

from src.inference.inference import create_predictor_from_file


class TestMissingValues:
    """Test handling of missing values"""

    @pytest.fixture
    def predictor(self):
        return create_predictor_from_file("tests/test_model.pkl")

    def test_missing_single_field(self, predictor):
        """Test prediction with one missing field"""
        customer = {
            "customerid": "TEST001",
            "age": 35,
            "gender": "Male",
            "tenure": 24,
            # 'usage_frequency': MISSING
            "support_calls": 3,
            "payment_delay": 5,
            "subscription_type": "Premium",
            "contract_length": "Annual",
            "total_spend": 1250.50,
            "last_interaction": 10,
        }

        # Should handle missing value gracefully
        try:
            result = predictor.predict(customer)
            assert "churn_probability" in result
        except Exception as e:
            # If it raises an exception, it should be informative
            assert "missing" in str(e).lower() or "required" in str(e).lower()

    def test_multiple_missing_fields(self, predictor):
        """Test prediction with multiple missing fields"""
        customer = {
            "customerid": "TEST001",
            "age": 35,
            "gender": "Male",
            # Several fields missing
        }

        try:
            result = predictor.predict(customer)
        except Exception as e:
            # Should provide meaningful error
            assert isinstance(e, (ValueError, KeyError, AttributeError))

    def test_none_values(self, predictor):
        """Test handling of None values"""
        customer = {
            "customerid": "TEST001",
            "age": None,
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

        # Should handle None values
        result = predictor.predict(customer)
        assert result is not None


class TestInvalidDataTypes:
    """Test handling of invalid data types"""

    @pytest.fixture
    def predictor(self):
        return create_predictor_from_file("tests/test_model.pkl")

    def test_string_for_numeric(self, predictor):
        """Test string value for numeric field"""
        customer = {
            "customerid": "TEST001",
            "age": "thirty-five",  # String instead of int
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

        try:
            result = predictor.predict(customer)
            # If it succeeds, the value should be handled
            assert result is not None
        except (ValueError, TypeError):
            # Or it should raise appropriate error
            pass

    def test_negative_values(self, predictor):
        """Test negative values where they shouldn't be"""
        customer = {
            "customerid": "TEST001",
            "age": -35,  # Negative age
            "gender": "Male",
            "tenure": -24,  # Negative tenure
            "usage_frequency": 15,
            "support_calls": 3,
            "payment_delay": 5,
            "subscription_type": "Premium",
            "contract_length": "Annual",
            "total_spend": 1250.50,
            "last_interaction": 10,
        }

        # Should still make prediction (or raise meaningful error)
        result = predictor.predict(customer)
        assert result is not None


class TestOutOfRangeValues:
    """Test handling of out-of-range values"""

    @pytest.fixture
    def predictor(self):
        return create_predictor_from_file("tests/test_model.pkl")

    def test_extreme_age(self, predictor):
        """Test with extremely high age"""
        customer = {
            "customerid": "TEST001",
            "age": 200,  # Unrealistic age
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

        result = predictor.predict(customer)
        assert "churn_probability" in result

    def test_extreme_spend(self, predictor):
        """Test with extremely high spending"""
        customer = {
            "customerid": "TEST001",
            "age": 35,
            "gender": "Male",
            "tenure": 24,
            "usage_frequency": 15,
            "support_calls": 3,
            "payment_delay": 5,
            "subscription_type": "Premium",
            "contract_length": "Annual",
            "total_spend": 9999999.99,  # Very high spend
            "last_interaction": 10,
        }

        result = predictor.predict(customer)
        assert "churn_probability" in result

    def test_zero_tenure(self, predictor):
        """Test with zero tenure"""
        customer = {
            "customerid": "TEST001",
            "age": 35,
            "gender": "Male",
            "tenure": 0,  # Zero tenure
            "usage_frequency": 15,
            "support_calls": 3,
            "payment_delay": 5,
            "subscription_type": "Premium",
            "contract_length": "Annual",
            "total_spend": 1250.50,
            "last_interaction": 10,
        }

        result = predictor.predict(customer)
        assert "churn_probability" in result


class TestEmptyInput:
    """Test handling of empty inputs"""

    @pytest.fixture
    def predictor(self):
        return create_predictor_from_file("tests/test_model.pkl")

    def test_empty_dict(self, predictor):
        """Test prediction with empty dict"""
        with pytest.raises(Exception):
            predictor.predict({})

    def test_empty_list_batch(self, predictor):
        """Test batch prediction with empty list"""
        result = predictor.predict_batch([])
        assert result["total_customers"] == 0


class TestMalformedInput:
    """Test handling of malformed input"""

    @pytest.fixture
    def predictor(self):
        return create_predictor_from_file("tests/test_model.pkl")

    def test_invalid_gender(self, predictor):
        """Test with invalid gender value"""
        customer = {
            "customerid": "TEST001",
            "age": 35,
            "gender": "Unknown",  # Invalid gender
            "tenure": 24,
            "usage_frequency": 15,
            "support_calls": 3,
            "payment_delay": 5,
            "subscription_type": "Premium",
            "contract_length": "Annual",
            "total_spend": 1250.50,
            "last_interaction": 10,
        }

        # Should handle gracefully
        result = predictor.predict(customer)
        assert result is not None

    def test_invalid_subscription_type(self, predictor):
        """Test with invalid subscription type"""
        customer = {
            "customerid": "TEST001",
            "age": 35,
            "gender": "Male",
            "tenure": 24,
            "usage_frequency": 15,
            "support_calls": 3,
            "payment_delay": 5,
            "subscription_type": "Diamond",  # Invalid type
            "contract_length": "Annual",
            "total_spend": 1250.50,
            "last_interaction": 10,
        }

        result = predictor.predict(customer)
        assert result is not None


class TestSpecialCharacters:
    """Test handling of special characters"""

    @pytest.fixture
    def predictor(self):
        return create_predictor_from_file("tests/test_model.pkl")

    def test_special_chars_in_customerid(self, predictor):
        """Test special characters in customer ID"""
        customer = {
            "customerid": "TEST@#$%001",
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

        result = predictor.predict(customer)
        assert result["customerid"] == "TEST@#$%001"


class TestBoundaryValues:
    """Test boundary values"""

    @pytest.fixture
    def predictor(self):
        return create_predictor_from_file("tests/test_model.pkl")

    def test_minimum_age(self, predictor):
        """Test with minimum reasonable age"""
        customer = {
            "customerid": "TEST001",
            "age": 18,
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

        result = predictor.predict(customer)
        assert "churn_probability" in result

    def test_all_zeros(self, predictor):
        """Test with all zero values"""
        customer = {
            "customerid": "TEST001",
            "age": 0,
            "gender": "Male",
            "tenure": 0,
            "usage_frequency": 0,
            "support_calls": 0,
            "payment_delay": 0,
            "subscription_type": "Basic",
            "contract_length": "Monthly",
            "total_spend": 0.0,
            "last_interaction": 0,
        }

        result = predictor.predict(customer)
        assert result is not None


class TestLargeBatch:
    """Test large batch processing"""

    @pytest.fixture
    def predictor(self):
        return create_predictor_from_file("tests/test_model.pkl")

    def test_large_batch_prediction(self, predictor):
        """Test prediction with large batch"""
        customers = [
            {
                "customerid": f"TEST{i:05d}",
                "age": 25 + (i % 50),
                "gender": "Male" if i % 2 == 0 else "Female",
                "tenure": 6 + (i % 60),
                "usage_frequency": 5 + (i % 30),
                "support_calls": i % 10,
                "payment_delay": i % 30,
                "subscription_type": ["Basic", "Premium", "Standard"][i % 3],
                "contract_length": ["Monthly", "Annual", "Quarterly"][i % 3],
                "total_spend": 100 + (i * 10),
                "last_interaction": 1 + (i % 30),
            }
            for i in range(1000)
        ]

        result = predictor.predict_batch(customers)

        assert result["total_customers"] == 1000
        assert len(result["predictions"]) == 1000


class TestDuplicateIds:
    """Test handling of duplicate customer IDs"""

    @pytest.fixture
    def predictor(self):
        return create_predictor_from_file("tests/test_model.pkl")

    def test_duplicate_customer_ids(self, predictor):
        """Test batch prediction with duplicate IDs"""
        customer_base = {
            "customerid": "DUPLICATE001",
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

        customers = [customer_base, {**customer_base, "age": 40}, customer_base.copy()]

        result = predictor.predict_batch(customers)
        assert len(result["predictions"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
