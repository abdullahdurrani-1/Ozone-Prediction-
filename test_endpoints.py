#!/usr/bin/env python
"""Quick test of API endpoints"""

import requests
import json

API_URL = 'http://localhost:8000'

# Test data
test_data = {
    'pressure': 950,
    'day_of_week': 3,
    'is_weekend': 0,
    'week_of_year': 15,
    'quarter': 2,
    'day_of_year': 100,
    'is_holiday_season': 0,
    'hour_sin': 0.5,
    'hour_cos': 0.866,
    'month_sin': 0.866,
    'month_cos': 0.5,
    'day_of_week_sin': 0.6,
    'day_of_week_cos': 0.8,
    'ozone_lag1': 45.2,
    'ozone_lag3': 44.8,
    'ozone_lag6': 43.5,
    'ozone_lag24': 40.2,
    'ozone_rolling_mean_3h': 45.0,
    'ozone_rolling_std_3h': 2.5,
    'ozone_rolling_mean_24h': 42.0,
    'pressure_month_interaction': 5700,
    'hour_pressure_interaction': 475,
    'pressure_squared': 902500,
    'pressure_cubed': 857375000,
    'site_mean_ozone': 45.0,
    'site_std_ozone': 8.0,
    'month_hour_avg_ozone': 45.0,
    'hour_avg_ozone': 45.0,
    'month_avg_ozone': 44.0,
    'ozone_deviation_from_site_mean': 0.2
}

print("=" * 60)
print("Testing Ozone API Endpoints")
print("=" * 60)

# Test 1: Health
print("\n1. Testing /health...")
try:
    r = requests.get(f'{API_URL}/health')
    print(f"   Status: {r.status_code}")
    print(f"   Response: {json.dumps(r.json(), indent=4)[:200]}...")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 2: Info
print("\n2. Testing /info...")
try:
    r = requests.get(f'{API_URL}/info')
    print(f"   Status: {r.status_code}")
    data = r.json()
    print(f"   Models available: ML={len(data.get('available_models', {}).get('ml_models', []))} DL={len(data.get('available_models', {}).get('dl_models', []))}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 3: Single Prediction
print("\n3. Testing /predict/random_forest...")
try:
    r = requests.post(f'{API_URL}/predict/random_forest', json=test_data)
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        print(f"   Prediction: {r.json()}")
    else:
        print(f"   Error: {r.text[:200]}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 4: Compare Models
print("\n4. Testing /compare-models...")
try:
    r = requests.post(f'{API_URL}/compare-models', json=test_data)
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        ml_count = len(data.get('ml_models', []))
        dl_count = len(data.get('dl_models', []))
        print(f"   ML Models: {ml_count}, DL Models: {dl_count}")
        if ml_count > 0:
            print(f"   First ML model: {data['ml_models'][0]}")
    else:
        print(f"   Error: {r.text[:200]}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 5: Batch Predict
print("\n5. Testing /batch-predict/random_forest...")
try:
    batch_data = {'samples': [test_data, test_data]}
    r = requests.post(f'{API_URL}/batch-predict/random_forest', json=batch_data)
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"   Total samples: {data.get('total_samples')}")
        print(f"   Successful: {data.get('successful_predictions')}")
        print(f"   Predictions: {len(data.get('predictions', []))}")
    else:
        print(f"   Error: {r.text[:200]}")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
