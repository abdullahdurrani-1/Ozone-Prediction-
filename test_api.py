"""
Test script for Ozone Prediction API
Run the API first: python app.py
Then: python test_api.py
"""
import requests
import json
import time
from typing import Dict, Any

API_URL = "http://localhost:8000"

# ANSI colors for output
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Load sample prediction data
with open("sample_request.json", "r") as f:
    sample_data = json.load(f)


def print_section(title: str):
    """Print a formatted section title"""
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}{title:^60}{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}\n")


def print_success(msg: str):
    """Print success message"""
    print(f"{GREEN}✓ {msg}{RESET}")


def print_error(msg: str):
    """Print error message"""
    print(f"{RED}✗ {msg}{RESET}")


def print_info(msg: str):
    """Print info message"""
    print(f"{BLUE}ℹ {msg}{RESET}")


def test_health():
    """Test health check endpoint"""
    print_section("Health Check")
    
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print_success(f"API is {data['status'].upper()}")
            print(f"  Version: {data['version']}")
            print(f"  Models: {data['models_loaded']}/{data['total_models']} loaded")
            print(f"  Google AI: {'Available' if data['google_ai_available'] else 'Not configured'}")
            return True
        else:
            print_error(f"Unexpected status: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Failed to connect: {str(e)}")
        return False


def test_info():
    """Test info endpoint"""
    print_section("API Information")
    
    try:
        response = requests.get(f"{API_URL}/info")
        if response.status_code == 200:
            data = response.json()
            print_success("Retrieved API information")
            print(f"  API: {data['api']['name']} v{data['api']['version']}")
            print(f"  Models: {data['models']['loaded']}/{data['models']['total']}")
            print(f"  Features: {data['features']['total_features']}")
            return True
        else:
            print_error(f"Failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False


def test_single_prediction(model_name: str):
    """Test single model prediction"""
    print(f"\n  Testing {model_name}...", end=" ")
    
    try:
        response = requests.post(
            f"{API_URL}/predict/{model_name}",
            json=sample_data,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"{GREEN}✓ {data['prediction']:.2f} ppbv (conf: {data['confidence']:.2f}){RESET}")
            return data
        else:
            print(f"{RED}✗ Status {response.status_code}{RESET}")
            return None
    except Exception as e:
        print(f"{RED}✗ Error: {str(e)}{RESET}")
        return None


def test_predictions():
    """Test all prediction endpoints"""
    print_section("Model Predictions")
    
    # Test individual models
    models_to_test = [
        "random_forest",
        "xgboost",
        "lstm",
        "gru"
    ]
    
    results = {}
    print(f"{YELLOW}Testing individual models...{RESET}")
    for model_name in models_to_test:
        result = test_single_prediction(model_name)
        if result:
            results[model_name] = result["prediction"]
    
    # Test predict-all
    print(f"\n{YELLOW}Testing predict-all endpoint...{RESET}")
    try:
        response = requests.post(f"{API_URL}/predict-all", json=sample_data, timeout=15)
        if response.status_code == 200:
            data = response.json()
            print_success(f"Got predictions from {len(data['ml_models']) + len(data['dl_models'])} models")
        else:
            print_error(f"Failed: {response.status_code}")
    except Exception as e:
        print_error(f"Error: {str(e)}")
    
    # Test ensemble
    print(f"\n{YELLOW}Testing ensemble prediction...{RESET}")
    try:
        response = requests.post(f"{API_URL}/predict-ensemble", json=sample_data, timeout=15)
        if response.status_code == 200:
            data = response.json()
            print_success(f"Ensemble prediction: {data['ensemble_prediction']:.2f} ppbv")
            print(f"  Models used: {data['models_used']}")
            print(f"  Std deviation: {data['std_deviation']:.2f}")
            print(f"  Confidence: {data['confidence']:.3f}")
        else:
            print_error(f"Failed: {response.status_code}")
    except Exception as e:
        print_error(f"Error: {str(e)}")


def test_comparisons():
    """Test comparison endpoint"""
    print_section("Model Comparison")
    
    try:
        response = requests.post(f"{API_URL}/compare-models", json=sample_data, timeout=30)
        if response.status_code == 200:
            data = response.json()
            stats = data['statistics']
            
            print_success("Model comparison completed")
            print(f"\n  ML Models: {len(data['comparison']['ml_models'])}")
            print(f"  DL Models: {len(data['comparison']['dl_models'])}")
            print(f"\n  Statistics:")
            print(f"    Min prediction: {stats['min_prediction']:.2f} ppbv")
            print(f"    Max prediction: {stats['max_prediction']:.2f} ppbv")
            print(f"    Avg prediction: {stats['avg_prediction']:.2f} ppbv")
            print(f"    Std deviation: {stats['std_deviation']:.2f}")
            return True
        else:
            print_error(f"Failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False


def test_explanations():
    """Test explanation endpoints with AI"""
    print_section("AI Explanations (Google Gemini)")
    
    models_to_explain = ["random_forest", "lstm"]
    
    for model_name in models_to_explain:
        print(f"\n{YELLOW}Getting explanation for {model_name}...{RESET}")
        try:
            response = requests.post(
                f"{API_URL}/explain/{model_name}",
                json=sample_data,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print_success(f"Prediction: {data['prediction']:.2f} ppbv")
                print(f"\n{YELLOW}Explanation:{RESET}")
                print(f"  {data['explanation'][:200]}...")
                print(f"\n{YELLOW}Recommendation:{RESET}")
                if "factors" in data and "recommendation" in data["factors"]:
                    print(f"  {data['factors']['recommendation']}")
            else:
                print_error(f"Failed: {response.status_code}")
        except requests.exceptions.Timeout:
            print_error("Request timed out (Google AI might be slow first time)")
        except Exception as e:
            print_error(f"Error: {str(e)}")


def test_batch_prediction():
    """Test batch prediction"""
    print_section("Batch Predictions")
    
    # Create multiple samples
    batch_request = {
        "samples": [sample_data, sample_data, sample_data]
    }
    
    try:
        response = requests.post(
            f"{API_URL}/batch-predict/random_forest",
            json=batch_request,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Batch processed: {data['successful_predictions']}/{data['total_samples']} samples")
            print(f"  Average confidence: {data['average_confidence']:.3f}")
        else:
            print_error(f"Failed: {response.status_code}")
    except Exception as e:
        print_error(f"Error: {str(e)}")


def main():
    """Run all tests"""
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}Ozone Prediction API - Test Suite{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}")
    
    print(f"\n{YELLOW}API URL: {API_URL}{RESET}")
    print(f"{YELLOW}Waiting for API to be ready...{RESET}\n")
    
    # Wait for API to be ready
    max_retries = 5
    for i in range(max_retries):
        try:
            requests.get(f"{API_URL}/health", timeout=2)
            break
        except:
            if i < max_retries - 1:
                time.sleep(2)
            else:
                print_error(f"Could not connect to API at {API_URL}")
                print_error("Make sure the API is running: python app.py")
                return
    
    # Run tests
    tests = [
        ("Health Check", test_health),
        ("API Information", test_info),
        ("Predictions", test_predictions),
        ("Model Comparisons", test_comparisons),
        ("Batch Predictions", test_batch_prediction),
        ("Explanations with AI", test_explanations),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result is not False:
                passed += 1
        except Exception as e:
            print_error(f"Test failed with exception: {str(e)}")
            failed += 1
    
    # Print summary
    print_section("Test Summary")
    total = len(tests)
    print_info(f"Tests completed: {passed}/{total} passed")
    
    if failed == 0:
        print(f"{GREEN}{BOLD}✓ All tests passed!{RESET}\n")
    else:
        print(f"{RED}{BOLD}✗ {failed} test(s) failed{RESET}\n")


if __name__ == "__main__":
    main()
