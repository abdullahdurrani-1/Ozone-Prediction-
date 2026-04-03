#!/usr/bin/env python3
"""
Ozone Prediction API - Complete System Validation
Run this to verify everything is working properly
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

# ANSI Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"

class APIValidator:
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = 0
        
    def print_header(self, title):
        print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
        print(f"{BOLD}{BLUE}{title:^60}{RESET}")
        print(f"{BOLD}{BLUE}{'='*60}{RESET}\n")
    
    def check_pass(self, msg):
        print(f"{GREEN}✓ {msg}{RESET}")
        self.checks_passed += 1
    
    def check_fail(self, msg):
        print(f"{RED}✗ {msg}{RESET}")
        self.checks_failed += 1
    
    def check_warn(self, msg):
        print(f"{YELLOW}⚠ {msg}{RESET}")
        self.warnings += 1
    
    def check_info(self, msg):
        print(f"{BLUE}ℹ {msg}{RESET}")
    
    # ==================== PHASE 1: Pre-flight Checks ====================
    def validate_python_version(self):
        """Check Python version"""
        self.print_header("Phase 1: Python & Environment")
        
        import platform
        py_version = platform.python_version()
        major, minor, _ = map(int, py_version.split('.'))
        
        if major == 3 and minor >= 9:
            self.check_pass(f"Python {py_version} (✓ compatible)")
        else:
            self.check_fail(f"Python {py_version} (need 3.9+)")
    
    def validate_dependencies(self):
        """Check required packages"""
        required = {
            'fastapi': 'FastAPI',
            'uvicorn': 'Uvicorn',
            'pydantic': 'Pydantic',
            'numpy': 'NumPy',
            'pandas': 'Pandas',
            'sklearn': 'Scikit-learn',
            'xgboost': 'XGBoost',
            'tensorflow': 'TensorFlow',
            'google.generativeai': 'Google AI',
            'requests': 'Requests'
        }
        
        self.check_info("Checking dependencies...")
        missing = []
        
        for module, name in required.items():
            try:
                __import__(module)
                self.check_pass(f"{name} installed")
            except ImportError:
                self.check_fail(f"{name} NOT installed - run: pip install -r requirements.txt")
                missing.append(module)
        
        return len(missing) == 0
    
    def validate_structure(self):
        """Check project structure"""
        self.check_info("Checking project structure...")
        
        required_files = [
            'app.py',
            'config.py',
            'schemas.py',
            'model_manager.py',
            'services.py',
            'requirements.txt',
            'Dockerfile',
            'docker-compose.yml',
            '.env.example'
        ]
        
        all_exist = True
        for file in required_files:
            if os.path.exists(file):
                self.check_pass(f"Found {file}")
            else:
                self.check_fail(f"Missing {file}")
                all_exist = False
        
        # Check models directory
        if os.path.exists('models'):
            self.check_pass("models/ directory exists")
        else:
            self.check_warn("models/ directory missing - create it: mkdir models")
            all_exist = False
        
        return all_exist
    
    def validate_env_config(self):
        """Check environment configuration"""
        self.check_info("Checking environment configuration...")
        
        if os.path.exists('.env'):
            self.check_pass(".env file exists")
            
            with open('.env', 'r') as f:
                content = f.read()
            
            if 'GOOGLE_API_KEY' in content:
                if 'GOOGLE_API_KEY=' in content and len(content.split('GOOGLE_API_KEY=')[1].split('\n')[0].strip()) > 10:
                    self.check_pass("GOOGLE_API_KEY is configured")
                else:
                    self.check_warn("GOOGLE_API_KEY is set but may be empty")
            else:
                self.check_warn("GOOGLE_API_KEY not found in .env")
        else:
            self.check_warn(".env file not found - run: cp .env.example .env")
    
    # ==================== PHASE 2: Model Files ====================
    def validate_models(self):
        """Check if model files exist"""
        self.print_header("Phase 2: Model Files")
        self.check_info("Checking trained models...")
        
        required_models = {
            'dummy_baseline.pkl': 'Dummy Baseline (ML)',
            'linear_regression.pkl': 'Linear Regression (ML)',
            'decision_tree.pkl': 'Decision Tree (ML)',
            'random_forest.pkl': 'Random Forest (ML)',
            'xgboost.pkl': 'XGBoost (ML)',
            'neural_network_model.h5': 'Neural Network (DL)',
            'lstm_model.h5': 'LSTM (DL)',
            'gru_model.h5': 'GRU (DL)',
            'scaler.pkl': 'Feature Scaler'
        }
        
        models_path = Path('models')
        models_found = 0
        
        for filename, description in required_models.items():
            filepath = models_path / filename
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                self.check_pass(f"{description:30} ({size_mb:.1f} MB)")
                models_found += 1
            else:
                self.check_warn(f"{description:30} missing")
        
        print()
        print(f"Models found: {models_found}/{len(required_models)}")
        
        if models_found < 6:
            self.check_warn("Only partial models available - some endpoints will fail")
        
        return models_found >= 6
    
    # ==================== PHASE 3: Import Tests ====================
    def validate_imports(self):
        """Test if core modules import correctly"""
        self.print_header("Phase 3: Module Imports")
        self.check_info("Testing module imports...")
        
        try:
            import fastapi
            self.check_pass("FastAPI imports successfully")
        except Exception as e:
            self.check_fail(f"FastAPI import failed: {str(e)}")
        
        try:
            from config import settings
            self.check_pass("config.py imports successfully")
        except Exception as e:
            self.check_fail(f"config.py import failed: {str(e)}")
        
        try:
            from schemas import PredictionRequest
            self.check_pass("schemas.py imports successfully")
        except Exception as e:
            self.check_fail(f"schemas.py import failed: {str(e)}")
        
        try:
            from model_manager import model_manager
            self.check_pass("model_manager.py imports successfully")
        except Exception as e:
            self.check_fail(f"model_manager.py import failed: {str(e)}")
        
        try:
            from services import prediction_service, explanation_service
            self.check_pass("services.py imports successfully")
        except Exception as e:
            self.check_fail(f"services.py import failed: {str(e)}")
    
    # ==================== PHASE 4: Model Loading ====================
    def validate_model_loading(self):
        """Test if models load correctly"""
        self.print_header("Phase 4: Model Loading")
        self.check_info("Loading models into memory...")
        
        try:
            from model_manager import model_manager
            
            if model_manager.models_loaded:
                self.check_pass("Models loaded successfully")
                
                loaded_ml = len(model_manager.ml_models)
                loaded_dl = len(model_manager.dl_models)
                
                print(f"\n  ML Models loaded: {loaded_ml}/5")
                print(f"  DL Models loaded: {loaded_dl}/3")
                print(f"  Total: {loaded_ml + loaded_dl}/8")
                
                if loaded_ml > 0:
                    print(f"  ML models: {', '.join(model_manager.ml_models.keys())}")
                
                if loaded_dl > 0:
                    print(f"  DL models: {', '.join(model_manager.dl_models.keys())}")
                
                if model_manager.scaler is None:
                    self.check_warn("Feature scaler not loaded")
                else:
                    self.check_pass("Feature scaler loaded")
            else:
                self.check_fail("Models failed to load")
        
        except Exception as e:
            self.check_fail(f"Error loading models: {str(e)}")
    
    # ==================== PHASE 5: API Startup ====================
    def validate_api_startup(self):
        """Test if API starts correctly"""
        self.print_header("Phase 5: API Startup Test")
        self.check_info("Starting API in background...")
        
        # Check if port is available
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port_available = sock.connect_ex(('127.0.0.1', 8000)) != 0
        sock.close()
        
        if not port_available:
            self.check_warn("Port 8000 is already in use - API might already be running")
            return True
        
        try:
            import uvicorn
            self.check_pass("Uvicorn can be imported")
        except:
            self.check_fail("Uvicorn import failed")
            return False
        
        # Try to run APP import
        try:
            from app import app
            self.check_pass("app.py imports and FastAPI app created")
            return True
        except Exception as e:
            self.check_fail(f"app.py failed to start: {str(e)}")
            return False
    
    # ==================== PHASE 6: API Endpoints ====================
    def validate_endpoints(self):
        """Test API endpoints"""
        self.print_header("Phase 6: API Endpoints")
        self.check_info("Testing API endpoints...")
        
        try:
            import requests
            
            base_url = "http://localhost:8000"
            
            # Wait for API to be ready
            print("\nWaiting for API to be ready...", end="")
            max_attempts = 10
            api_ready = False
            
            for i in range(max_attempts):
                try:
                    response = requests.get(f"{base_url}/health", timeout=2)
                    if response.status_code == 200:
                        api_ready = True
                        print(f" {GREEN}✓{RESET}\n")
                        break
                except:
                    print(".", end="", flush=True)
                    time.sleep(1)
            
            if not api_ready:
                self.check_fail("API did not start in time - run: python app.py")
                return False
            
            # Test health endpoint
            response = requests.get(f"{base_url}/health")
            if response.status_code == 200:
                data = response.json()
                self.check_pass(f"GET /health → Status: {data['status']}")
                self.check_info(f"  Models: {data['models_loaded']}/{data['total_models']}")
                self.check_info(f"  Google AI: {'Available' if data['google_ai_available'] else 'Not configured'}")
            else:
                self.check_fail(f"GET /health failed: {response.status_code}")
            
            # Test info endpoint
            response = requests.get(f"{base_url}/info")
            if response.status_code == 200:
                self.check_pass("GET /info → Retrieved API information")
            else:
                self.check_fail(f"GET /info failed: {response.status_code}")
            
            # Load sample data
            with open('sample_request.json', 'r') as f:
                sample_data = json.load(f)
            
            # Test prediction endpoint
            response = requests.post(f"{base_url}/predict/random_forest", json=sample_data)
            if response.status_code == 200:
                data = response.json()
                self.check_pass(f"POST /predict/random_forest → Prediction: {data['prediction']:.2f} ppbv")
            else:
                self.check_warn(f"POST /predict/random_forest → {response.status_code}")
            
            # Test ensemble endpoint
            response = requests.post(f"{base_url}/predict-ensemble", json=sample_data)
            if response.status_code == 200:
                data = response.json()
                self.check_pass(f"POST /predict-ensemble → {data['ensemble_prediction']:.2f} ppbv (models: {data['models_used']})")
            else:
                self.check_warn(f"POST /predict-ensemble → {response.status_code}")
            
            # Test explanation endpoint
            response = requests.post(f"{base_url}/explain/random_forest", json=sample_data)
            if response.status_code == 200:
                data = response.json()
                self.check_pass(f"POST /explain/random_forest → Explanation provided")
            else:
                self.check_warn(f"POST /explain/random_forest → {response.status_code}")
            
            return True
        
        except requests.exceptions.ConnectionError:
            self.check_fail("Cannot connect to API on http://localhost:8000")
            self.check_info("Make sure API is running: python app.py")
            return False
        except Exception as e:
            self.check_fail(f"Endpoint testing failed: {str(e)}")
            return False
    
    # ==================== SUMMARY ====================
    def print_summary(self):
        """Print validation summary"""
        self.print_header("Validation Summary")
        
        total = self.checks_passed + self.checks_failed
        print(f"Checks passed: {GREEN}{self.checks_passed}{RESET}")
        print(f"Checks failed: {RED}{self.checks_failed}{RESET}")
        print(f"Warnings: {YELLOW}{self.warnings}{RESET}")
        print(f"Total: {total}\n")
        
        if self.checks_failed == 0 and self.warnings == 0:
            print(f"{BOLD}{GREEN}✓ All systems operational! 🎉{RESET}\n")
            print("You can now:")
            print("  • Deploy with: docker-compose up -d")
            print("  • Run tests with: python test_api.py")
            print("  • Visit: http://localhost:8000/docs")
            return True
        elif self.checks_failed == 0:
            print(f"{BOLD}{YELLOW}⚠ System operational with warnings{RESET}\n")
            print("Review warnings above and fix before production deployment")
            return True
        else:
            print(f"{BOLD}{RED}✗ System has critical issues{RESET}\n")
            print("Fix failures above before proceeding")
            return False
    
    def run_full_validation(self):
        """Run complete validation"""
        print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
        print(f"{BOLD}{BLUE}Ozone Prediction API - System Validator{RESET}")
        print(f"{BOLD}{BLUE}{'='*60}{RESET}")
        
        # Phase 1
        self.validate_python_version()
        deps_ok = self.validate_dependencies()
        struct_ok = self.validate_structure()
        self.validate_env_config()
        
        if not deps_ok:
            self.check_fail("Dependencies missing - stopping validation")
            self.print_summary()
            return False
        
        # Phase 2
        self.validate_models()
        
        # Phase 3
        self.validate_imports()
        
        # Phase 4
        self.validate_model_loading()
        
        # Phase 5
        api_ok = self.validate_api_startup()
        
        # Phase 6 (only if API started)
        if api_ok:
            self.validate_endpoints()
        
        # Summary
        return self.print_summary()


if __name__ == "__main__":
    validator = APIValidator()
    success = validator.run_full_validation()
    sys.exit(0 if success else 1)
