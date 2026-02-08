"""
Test S3 Model Integration
==========================
Simple script to test S3 model upload/download functionality.

Usage:
    python scripts/test_s3_integration.py
"""

import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_s3_connection():
    """Test basic S3 connection."""
    logger.info("=" * 80)
    logger.info("TEST 1: S3 Connection")
    logger.info("=" * 80)

    try:
        from src.utils.s3_handler import S3ModelHandler

        bucket_name = os.getenv("S3_BUCKET_NAME")
        if not bucket_name:
            logger.error("S3_BUCKET_NAME environment variable not set")
            return False

        handler = S3ModelHandler(bucket_name=bucket_name)
        logger.info(f"✓ Connected to S3 bucket: {bucket_name}")

        # List models
        models = handler.list_models()
        logger.info(f"✓ Found {len(models)} models in bucket")

        if models:
            for model in models[:5]:  # Show first 5
                size_mb = model["size"] / 1024 / 1024
                logger.info(f"  - {model['name']} ({size_mb:.2f} MB)")

        return True

    except Exception as e:
        logger.error(f"✗ S3 connection failed: {e}")
        return False


def test_model_download():
    """Test downloading a model from S3."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Model Download")
    logger.info("=" * 80)

    try:
        from src.utils.s3_handler import S3ModelHandler

        bucket_name = os.getenv("S3_BUCKET_NAME")
        model_name = os.getenv("S3_MODEL_NAME")

        if not model_name:
            logger.warning("S3_MODEL_NAME not set, skipping download test")
            return True

        handler = S3ModelHandler(bucket_name=bucket_name)

        # Check if model exists
        if not handler.model_exists(model_name):
            logger.error(f"✗ Model not found: {model_name}")
            logger.info("Available models:")
            for model in handler.list_models():
                logger.info(f"  - {model['name']}")
            return False

        logger.info(f"✓ Model exists: {model_name}")

        # Get model info
        info = handler.get_model_info(model_name)
        logger.info(f"✓ Model size: {info['size'] / 1024 / 1024:.2f} MB")
        logger.info(f"✓ Last modified: {info['last_modified']}")

        # Download model
        logger.info("Downloading model...")
        model, metadata = handler.download_model(model_name)
        logger.info(f"✓ Model downloaded successfully")
        logger.info(f"✓ Model type: {type(model).__name__}")

        if metadata:
            logger.info("✓ Metadata:")
            for key, value in list(metadata.items())[:5]:
                logger.info(f"  - {key}: {value}")

        return True

    except Exception as e:
        logger.error(f"✗ Model download failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def test_inference_integration():
    """Test loading model through inference module."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Inference Integration")
    logger.info("=" * 80)

    try:
        from src.inference.inference import create_predictor_from_s3

        model_name = os.getenv("S3_MODEL_NAME")
        bucket_name = os.getenv("S3_BUCKET_NAME")

        if not model_name:
            logger.warning("S3_MODEL_NAME not set, skipping inference test")
            return True

        logger.info("Creating predictor from S3...")
        predictor = create_predictor_from_s3(
            model_name=model_name,
            bucket_name=bucket_name,
        )

        logger.info("✓ Predictor created successfully")

        # Get model info
        model_info = predictor.get_model_info()
        logger.info(f"✓ Model source: {model_info['source']}")
        logger.info(f"✓ Features: {model_info['feature_count']}")
        logger.info(f"✓ Model type: {model_info['model_type']}")

        # Test prediction
        logger.info("\nTesting prediction...")
        test_data = {
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

        result = predictor.predict(test_data)
        logger.info("✓ Prediction successful")
        logger.info(f"  - Churn probability: {result['churn_probability']}")
        logger.info(f"  - Churn prediction: {result['churn_prediction']}")
        logger.info(f"  - Risk level: {result['risk_level']}")

        return True

    except Exception as e:
        logger.error(f"✗ Inference integration failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 80)
    logger.info("S3 MODEL INTEGRATION TEST SUITE")
    logger.info("=" * 80)

    # Check environment variables
    required_vars = ["S3_BUCKET_NAME"]
    optional_vars = ["S3_MODEL_NAME", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]

    logger.info("\nEnvironment Configuration:")
    for var in required_vars:
        value = os.getenv(var)
        if value:
            logger.info(f"✓ {var}: {value}")
        else:
            logger.error(f"✗ {var}: NOT SET (REQUIRED)")

    for var in optional_vars:
        value = os.getenv(var)
        if value:
            # Mask credentials
            if "SECRET" in var or "KEY" in var:
                masked = (
                    value[:4] + "*" * (len(value) - 8) + value[-4:]
                    if len(value) > 8
                    else "****"
                )
                logger.info(f"✓ {var}: {masked}")
            else:
                logger.info(f"✓ {var}: {value}")
        else:
            logger.warning(f"⚠ {var}: NOT SET (optional)")

    # Run tests
    results = []

    results.append(("S3 Connection", test_s3_connection()))
    results.append(("Model Download", test_model_download()))
    results.append(("Inference Integration", test_inference_integration()))

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info("\n" + "-" * 80)
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info("=" * 80)

    if passed == total:
        logger.info("\n🎉 All tests passed! S3 integration is working correctly.")
        return 0
    else:
        logger.error(
            f"\n❌ {total - passed} test(s) failed. Please check the errors above."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
