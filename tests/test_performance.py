"""
Performance and load testing for inference engine
"""

import time

import numpy as np
import pytest

from src.inference.inference import create_predictor_from_file


class TestPredictionLatency:
    """Test prediction latency"""

    @pytest.fixture
    def predictor(self):
        return create_predictor_from_file("tests/test_model.pkl")

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

    def test_single_prediction_latency(self, predictor, sample_customer, benchmark):
        """Benchmark single prediction latency"""
        result = benchmark(predictor.predict, sample_customer)
        assert "churn_probability" in result

    def test_prediction_under_100ms(self, predictor, sample_customer):
        """Test that single prediction completes under 100ms"""
        start = time.time()
        result = predictor.predict(sample_customer)
        end = time.time()

        latency = (end - start) * 1000  # Convert to ms
        assert latency < 100, f"Prediction took {latency:.2f}ms, should be under 100ms"


class TestBatchThroughput:
    """Test batch processing throughput"""

    @pytest.fixture
    def predictor(self):
        return create_predictor_from_file("tests/test_model.pkl")

    @pytest.fixture
    def batch_customers(self):
        return [
            {
                "customerid": f"TEST{i:03d}",
                "age": 30 + (i % 40),
                "gender": "Male" if i % 2 == 0 else "Female",
                "tenure": 12 + (i % 48),
                "usage_frequency": 10 + (i % 20),
                "support_calls": 2 + (i % 8),
                "payment_delay": i % 30,
                "subscription_type": ["Basic", "Premium", "Standard"][i % 3],
                "contract_length": ["Monthly", "Annual", "Quarterly"][i % 3],
                "total_spend": 500 + (i * 50),
                "last_interaction": 5 + (i % 25),
            }
            for i in range(100)
        ]

    def test_batch_prediction_latency(self, predictor, batch_customers, benchmark):
        """Benchmark batch prediction throughput"""
        result = benchmark(predictor.predict_batch, batch_customers)
        assert result["total_customers"] == len(batch_customers)

    def test_batch_faster_than_sequential(self, predictor, batch_customers):
        """Test that batch prediction is faster than sequential"""
        # Batch prediction
        batch_start = time.time()
        batch_result = predictor.predict_batch(batch_customers)
        batch_end = time.time()
        batch_time = batch_end - batch_start

        # Sequential prediction (first 10 only)
        sample_customers = batch_customers[:10]
        sequential_start = time.time()
        for customer in sample_customers:
            predictor.predict(customer)
        sequential_end = time.time()
        sequential_time = sequential_end - sequential_start

        # Extrapolate sequential time for full batch
        extrapolated_time = (sequential_time / 10) * 100

        assert batch_time < extrapolated_time, "Batch should be faster than sequential"

    def test_large_batch_throughput(self, predictor):
        """Test throughput with large batch (1000 customers)"""
        large_batch = [
            {
                "customerid": f"TEST{i:04d}",
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

        start = time.time()
        result = predictor.predict_batch(large_batch)
        end = time.time()

        elapsed = end - start
        throughput = 1000 / elapsed  # predictions per second

        assert throughput > 10, (
            f"Throughput is {throughput:.2f} predictions/sec, should be >10"
        )


class TestMemoryUsage:
    """Test memory efficiency"""

    @pytest.fixture
    def predictor(self):
        return create_predictor_from_file("tests/test_model.pkl")

    def test_large_batch_memory(self, predictor):
        """Test that large batches don't cause memory issues"""
        # Create very large batch
        large_batch = [
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
            for i in range(5000)
        ]

        # Should complete without memory error
        result = predictor.predict_batch(large_batch)
        assert result["total_customers"] == 5000


class TestConcurrency:
    """Test concurrent prediction handling"""

    @pytest.fixture
    def predictor(self):
        return create_predictor_from_file("tests/test_model.pkl")

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

    def test_sequential_predictions(self, predictor, sample_customer):
        """Test multiple sequential predictions"""
        results = []
        for i in range(100):
            customer = {**sample_customer, "customerid": f"TEST{i:03d}"}
            result = predictor.predict(customer)
            results.append(result)

        assert len(results) == 100
        assert all("churn_probability" in r for r in results)


class TestStressTest:
    """Stress testing"""

    @pytest.fixture
    def predictor(self):
        return create_predictor_from_file("tests/test_model.pkl")

    def test_many_sequential_predictions(self, predictor):
        """Test many sequential predictions"""
        for i in range(1000):
            customer = {
                "customerid": f"STRESS{i:04d}",
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

            result = predictor.predict(customer)
            assert "churn_probability" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
