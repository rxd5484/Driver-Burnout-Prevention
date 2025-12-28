"""
Example API Usage for Driver Burnout Prevention System
Demonstrates how to interact with the FastAPI endpoints
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test API health endpoint"""
    print("="*60)
    print("TESTING HEALTH ENDPOINT")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def predict_single_driver():
    """Predict burnout risk for a high-risk driver"""
    print("="*60)
    print("PREDICTING HIGH-RISK DRIVER")
    print("="*60)
    
    # Example high-risk driver pattern
    high_risk_driver = {
        "driver_id": 101,
        "tenure_days": 150,
        "total_shifts": 120,
        "shifts_per_week": 5.5,
        "avg_weekly_hours": 55.0,  # Overworking
        "avg_shift_duration": 11.5,  # Long shifts
        "max_shift_duration": 14.0,
        "long_shifts_pct": 0.70,  # 70% long shifts
        "max_consecutive_days": 12,  # No breaks
        "avg_consecutive_days": 7.5,
        "late_night_shifts_pct": 0.45,
        "total_late_night_shifts": 35,
        "weekend_shifts_pct": 0.40,
        "avg_earnings_per_hour": 28.50,
        "earnings_std": 8.20,
        "earnings_trend": -1,  # Declining
        "earnings_decline_pct": -0.18,  # 18% decline
        "avg_break_duration": 12.0,  # Short breaks
        "breaks_per_shift": 1.2,  # Insufficient breaks
        "short_break_pct": 0.65,
        "shift_duration_variability": 2.8,
        "shifts_last_30d": 22,
        "hours_last_30d": 250.0,  # Way too much
        "current_rating": 4.3,
        "work_experience_years": 2.5
    }
    
    response = requests.post(f"{BASE_URL}/predict_burnout", json=high_risk_driver)
    result = response.json()
    
    print(f"\nDriver ID: {result['driver_id']}")
    print(f"Burnout Probability: {result['burnout_probability']:.2%}")
    print(f"Risk Level: {result['risk_level'].upper()}")
    
    print(f"\nTop Risk Factors:")
    for i, factor in enumerate(result['top_risk_factors'][:5], 1):
        print(f"  {i}. {factor['feature']}: {factor['value']:.2f} (importance: {factor['importance']:.4f})")
    
    print(f"\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  • {rec}")
    print()

def predict_healthy_driver():
    """Predict burnout risk for a healthy driver"""
    print("="*60)
    print("PREDICTING HEALTHY DRIVER")
    print("="*60)
    
    # Example healthy driver pattern
    healthy_driver = {
        "driver_id": 102,
        "tenure_days": 180,
        "total_shifts": 90,
        "shifts_per_week": 3.5,
        "avg_weekly_hours": 38.0,
        "avg_shift_duration": 8.5,
        "max_shift_duration": 10.0,
        "long_shifts_pct": 0.15,
        "max_consecutive_days": 5,
        "avg_consecutive_days": 3.2,
        "late_night_shifts_pct": 0.10,
        "total_late_night_shifts": 8,
        "weekend_shifts_pct": 0.25,
        "avg_earnings_per_hour": 32.00,
        "earnings_std": 4.50,
        "earnings_trend": 1,  # Increasing
        "earnings_decline_pct": 0.08,
        "avg_break_duration": 22.0,
        "breaks_per_shift": 2.5,
        "short_break_pct": 0.15,
        "shift_duration_variability": 1.2,
        "shifts_last_30d": 14,
        "hours_last_30d": 120.0,
        "current_rating": 4.8,
        "work_experience_years": 3.0
    }
    
    response = requests.post(f"{BASE_URL}/predict_burnout", json=healthy_driver)
    result = response.json()
    
    print(f"\nDriver ID: {result['driver_id']}")
    print(f"Burnout Probability: {result['burnout_probability']:.2%}")
    print(f"Risk Level: {result['risk_level'].upper()}")
    
    print(f"\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  • {rec}")
    print()

def get_interventions():
    """Get detailed intervention recommendations"""
    print("="*60)
    print("INTERVENTION RECOMMENDATIONS")
    print("="*60)
    
    critical_driver = {
        "driver_id": 103,
        "tenure_days": 90,
        "total_shifts": 85,
        "shifts_per_week": 6.5,
        "avg_weekly_hours": 65.0,  # Extremely overworked
        "avg_shift_duration": 12.0,
        "max_shift_duration": 15.0,
        "long_shifts_pct": 0.85,
        "max_consecutive_days": 18,  # Critical
        "avg_consecutive_days": 9.5,
        "late_night_shifts_pct": 0.60,  # Too many late nights
        "total_late_night_shifts": 50,
        "weekend_shifts_pct": 0.55,
        "avg_earnings_per_hour": 25.00,  # Low and declining
        "earnings_std": 12.00,
        "earnings_trend": -1,
        "earnings_decline_pct": -0.25,
        "avg_break_duration": 8.0,  # Very short breaks
        "breaks_per_shift": 0.8,  # Almost no breaks
        "short_break_pct": 0.90,
        "shift_duration_variability": 3.5,
        "shifts_last_30d": 28,
        "hours_last_30d": 340.0,
        "current_rating": 3.9,
        "work_experience_years": 1.5
    }
    
    response = requests.post(f"{BASE_URL}/intervention_recommendations", json=critical_driver)
    interventions = response.json()
    
    for intervention in interventions:
        print(f"\n{'='*60}")
        print(f"PRIORITY: {intervention['priority'].upper()}")
        print(f"Type: {intervention['recommendation_type']}")
        print(f"{'='*60}")
        print(f"\n{intervention['message']}\n")
        print("Action Items:")
        for action in intervention['action_items']:
            print(f"  ✓ {action}")
    print()

def batch_predict():
    """Batch prediction for multiple drivers"""
    print("="*60)
    print("BATCH PREDICTION")
    print("="*60)
    
    drivers = [
        {
            "driver_id": 201,
            "tenure_days": 120,
            "total_shifts": 80,
            "shifts_per_week": 4.5,
            "avg_weekly_hours": 42.0,
            "avg_shift_duration": 9.0,
            "max_shift_duration": 11.0,
            "long_shifts_pct": 0.35,
            "max_consecutive_days": 6,
            "avg_consecutive_days": 4.0,
            "late_night_shifts_pct": 0.20,
            "total_late_night_shifts": 15,
            "weekend_shifts_pct": 0.30,
            "avg_earnings_per_hour": 30.00,
            "earnings_std": 5.00,
            "earnings_trend": 0,
            "earnings_decline_pct": 0.02,
            "avg_break_duration": 18.0,
            "breaks_per_shift": 2.0,
            "short_break_pct": 0.30,
            "shift_duration_variability": 1.8,
            "shifts_last_30d": 18,
            "hours_last_30d": 160.0,
            "current_rating": 4.6,
            "work_experience_years": 2.0
        },
        {
            "driver_id": 202,
            "tenure_days": 200,
            "total_shifts": 150,
            "shifts_per_week": 5.0,
            "avg_weekly_hours": 52.0,
            "avg_shift_duration": 10.5,
            "max_shift_duration": 13.0,
            "long_shifts_pct": 0.60,
            "max_consecutive_days": 10,
            "avg_consecutive_days": 6.5,
            "late_night_shifts_pct": 0.35,
            "total_late_night_shifts": 42,
            "weekend_shifts_pct": 0.45,
            "avg_earnings_per_hour": 27.50,
            "earnings_std": 9.00,
            "earnings_trend": -1,
            "earnings_decline_pct": -0.15,
            "avg_break_duration": 13.5,
            "breaks_per_shift": 1.5,
            "short_break_pct": 0.55,
            "shift_duration_variability": 2.5,
            "shifts_last_30d": 24,
            "hours_last_30d": 260.0,
            "current_rating": 4.2,
            "work_experience_years": 3.5
        }
    ]
    
    response = requests.post(f"{BASE_URL}/batch_predict", json=drivers)
    result = response.json()
    
    print(f"\nTotal Drivers Analyzed: {result['total_drivers']}\n")
    
    for pred in result['predictions']:
        print(f"Driver {pred['driver_id']}: {pred['burnout_probability']:.2%} - {pred['risk_level'].upper()}")
    print()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("DRIVER BURNOUT PREVENTION API - EXAMPLES")
    print("="*60)
    print("\nNOTE: Make sure the API is running at http://localhost:8000")
    print("Start it with: python src/api/main.py\n")
    
    try:
        test_health()
        predict_single_driver()
        predict_healthy_driver()
        get_interventions()
        batch_predict()
        
        print("="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print("Please start the API server first:")
        print("  cd src && python api/main.py\n")
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
