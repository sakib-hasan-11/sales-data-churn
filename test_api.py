"""
Test script for the Churn Prediction API.
Run this script to verify the API is working correctly.
"""

import requests
import json
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

# Colors for console output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_test(name: str):
    """Print test name."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Test: {name}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")


def print_success(message: str):
    """Print success message."""
    print(f"{GREEN}✓ {message}{RESET}")


def print_error(message: str):
    """Print error message."""
    print(f"{RED}✗ {message}{RESET}")


def print_info(message: str):
    """Print info message."""
    print(f"{YELLOW}ℹ {message}{RESET}")


def test_root():
    """Test root endpoint."""
    print_test("Root Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        response.raise_for_status()
        data = response.json()
        
        print_info(f"Response: {json.dumps(data, indent=2)}")
        print_success("Root endpoint working")
        return True
    except Exception as e:
        print_error(f"Root endpoint failed: {e}")
        return False


def test_health():
    """Test health endpoint."""
    print_test("Health Check Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        
        print_info(f"Response: {json.dumps(data, indent=2)}")
        
        if data.get("model_loaded"):
            print_success("Health check passed - Model loaded")
        else:
            print_error("Health check passed but model not loaded")
        
        return True
    except Exception as e:
        print_error(f"Health check failed: {e}")
        return False


def test_model_info():
    """Test model info endpoint."""
    print_test("Model Info Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        response.raise_for_status()
        data = response.json()
        
        print_info(f"Response: {json.dumps(data, indent=2)}")
        print_success("Model info endpoint working")
        return True
    except Exception as e:
        print_error(f"Model info failed: {e}")
        return False


def test_single_prediction():
    """Test single prediction endpoint."""
    print_test("Single Prediction")
    
    customer_data = {
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
        "last_interaction": 10
    }
    
    try:
        print_info(f"Request: {json.dumps(customer_data, indent=2)}")
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json=customer_data,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()
        
        print_info(f"Response: {json.dumps(data, indent=2)}")
        
        # Validate response
        required_fields = ["churn_probability", "churn_prediction", "risk_level"]
        if all(field in data for field in required_fields):
            print_success(f"Prediction: {data['churn_prediction']} (Probability: {data['churn_probability']}, Risk: {data['risk_level']})")
            return True
        else:
            print_error("Response missing required fields")
            return False
            
    except Exception as e:
        print_error(f"Single prediction failed: {e}")
        return False


def test_high_risk_prediction():
    """Test prediction for high-risk customer."""
    print_test("High Risk Customer Prediction")
    
    high_risk_customer = {
        "customerid": "TEST002",
        "age": 28,
        "gender": "Female",
        "tenure": 3,
        "usage_frequency": 2,
        "support_calls": 12,
        "payment_delay": 25,
        "subscription_type": "Basic",
        "contract_length": "Monthly",
        "total_spend": 150.0,
        "last_interaction": 45
    }
    
    try:
        print_info(f"Request: {json.dumps(high_risk_customer, indent=2)}")
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json=high_risk_customer,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()
        
        print_info(f"Response: {json.dumps(data, indent=2)}")
        print_success(f"High-risk prediction: {data['churn_prediction']} (Probability: {data['churn_probability']}, Risk: {data['risk_level']})")
        return True
            
    except Exception as e:
        print_error(f"High-risk prediction failed: {e}")
        return False


def test_batch_prediction():
    """Test batch prediction endpoint."""
    print_test("Batch Prediction")
    
    batch_data = {
        "customers": [
            {
                "customerid": "BATCH001",
                "age": 35,
                "gender": "Male",
                "tenure": 24,
                "usage_frequency": 15,
                "support_calls": 3,
                "payment_delay": 5,
                "subscription_type": "Premium",
                "contract_length": "Annual",
                "total_spend": 1250.50,
                "last_interaction": 10
            },
            {
                "customerid": "BATCH002",
                "age": 28,
                "gender": "Female",
                "tenure": 6,
                "usage_frequency": 8,
                "support_calls": 7,
                "payment_delay": 15,
                "subscription_type": "Basic",
                "contract_length": "Monthly",
                "total_spend": 350.0,
                "last_interaction": 25
            },
            {
                "customerid": "BATCH003",
                "age": 45,
                "gender": "Male",
                "tenure": 48,
                "usage_frequency": 20,
                "support_calls": 1,
                "payment_delay": 0,
                "subscription_type": "Premium",
                "contract_length": "Annual",
                "total_spend": 2500.0,
                "last_interaction": 5
            }
        ]
    }
    
    try:
        print_info(f"Batch size: {len(batch_data['customers'])}")
        
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json=batch_data,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()
        
        print_info(f"Total customers: {data['total_customers']}")
        print_info(f"High risk count: {data['high_risk_count']}")
        print_info(f"Churn rate: {data['churn_rate']}")
        
        for pred in data['predictions']:
            print_info(f"  {pred['customerid']}: {pred['risk_level']} (prob={pred['churn_probability']})")
        
        print_success("Batch prediction working")
        return True
            
    except Exception as e:
        print_error(f"Batch prediction failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Churn Prediction API Test Suite{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    tests = [
        ("Root Endpoint", test_root),
        ("Health Check", test_health),
        ("Model Info", test_model_info),
        ("Single Prediction", test_single_prediction),
        ("High Risk Prediction", test_high_risk_prediction),
        ("Batch Prediction", test_batch_prediction),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print_error(f"Test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Test Summary{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"{status} - {name}")
    
    print(f"\n{BLUE}Total: {passed}/{total} tests passed{RESET}")
    
    if passed == total:
        print(f"{GREEN}All tests passed! ✓{RESET}")
        return 0
    else:
        print(f"{RED}Some tests failed! ✗{RESET}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(run_all_tests())
