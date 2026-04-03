import requests
import json

API_URL = 'http://localhost:8000'

data = {
    'pressure': 950,
    'day_of_week': 3,
    'is_weekend': 0,
    'week_of_year': 15,
    'quarter': 2,
    'day_of_year': 180,
    'is_holiday_season': 0,
    'hour_sin': 0.5,
    'hour_cos': 0.866,
    'month_sin': 0.866,
    'month_cos': 0.5,
    'day_of_week_sin': 0.6,
    'day_of_week_cos': 0.8,
    'ozone_lag1': 45.2,
    'ozone_lag3': 44.8,
    'ozone_lag6': 43.8,
    'ozone_lag24': 42.8,
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

models = ['random_forest', 'xgboost', 'neural_network', 'lstm', 'gru', 'linear_regression']

print('='*70)
print('TESTING ALL MODELS WITH FALLBACK')
print('='*70)

for model in models:
    try:
        resp = requests.post(f'{API_URL}/predict/{model}', json=data, timeout=5)
        if resp.status_code == 200:
            pred = resp.json()['prediction']
            status = '✅ GOOD' if pred < 55 else '⚠️  MODERATE' if pred < 75 else '🔴 UNHEALTHY'
            print(f'{model:20} -> {pred:6.2f} ppbv ({status})')
        else:
            print(f'{model:20} -> Error HTTP {resp.status_code}')
    except Exception as e:
        print(f'{model:20} -> Failed: {str(e)[:40]}')

print('='*70)
