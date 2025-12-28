"""
FastAPI Backend for Driver Burnout Prevention System
Real-time burnout risk scoring and intervention recommendations
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.burnout_predictor import BurnoutPredictor

app = FastAPI(
    title="Driver Burnout Prevention API",
    description="Real-time burnout risk assessment and intervention recommendations for rideshare drivers",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
predictor = BurnoutPredictor()
try:
    predictor.load_model('../models/')
    print("✓ Burnout prediction model loaded successfully")
except Exception as e:
    print(f"⚠ Warning: Could not load model - {e}")
    print("  Run src/models/burnout_predictor.py first to train the model")


# Pydantic models for API
class DriverWorkPattern(BaseModel):
    """Current work pattern metrics for a driver"""
    driver_id: int
    tenure_days: int = Field(..., description="Days since driver signup")
    total_shifts: int = Field(..., description="Total shifts completed")
    shifts_per_week: float = Field(..., description="Average shifts per week")
    avg_weekly_hours: float = Field(..., description="Average hours worked per week")
    avg_shift_duration: float = Field(..., description="Average shift duration in hours")
    max_shift_duration: float = Field(..., description="Longest shift duration")
    long_shifts_pct: float = Field(..., description="Percentage of shifts >10 hours")
    max_consecutive_days: int = Field(..., description="Maximum consecutive days worked")
    avg_consecutive_days: float = Field(..., description="Average consecutive days worked")
    late_night_shifts_pct: float = Field(..., description="Percentage of late-night shifts (11pm-5am)")
    total_late_night_shifts: int = Field(..., description="Total late-night shifts")
    weekend_shifts_pct: float = Field(..., description="Percentage of weekend shifts")
    avg_earnings_per_hour: float = Field(..., description="Average hourly earnings")
    earnings_std: float = Field(..., description="Earnings variability")
    earnings_trend: int = Field(..., description="Earnings trend (-1: declining, 0: stable, 1: increasing)")
    earnings_decline_pct: float = Field(..., description="Recent vs historical earnings decline")
    avg_break_duration: float = Field(..., description="Average break duration in minutes")
    breaks_per_shift: float = Field(..., description="Average breaks per shift")
    short_break_pct: float = Field(..., description="Percentage of breaks <15 minutes")
    shift_duration_variability: float = Field(..., description="Shift duration standard deviation")
    shifts_last_30d: int = Field(..., description="Shifts in last 30 days")
    hours_last_30d: float = Field(..., description="Hours worked in last 30 days")
    current_rating: float = Field(..., description="Current driver rating (1-5)")
    work_experience_years: float = Field(..., description="Years of rideshare experience")


class BurnoutRiskResponse(BaseModel):
    """Burnout risk assessment response"""
    driver_id: int
    burnout_probability: float
    risk_level: str  # low, medium, high, critical
    top_risk_factors: List[dict]
    recommendations: List[str]


class InterventionRecommendation(BaseModel):
    """Personalized intervention suggestion"""
    recommendation_type: str
    priority: str
    message: str
    action_items: List[str]


@app.get("/")
def root():
    """API health check"""
    return {
        "status": "healthy",
        "service": "Driver Burnout Prevention API",
        "version": "1.0.0",
        "endpoints": [
            "/predict_burnout",
            "/high_risk_drivers",
            "/intervention_recommendations",
            "/health"
        ]
    }


@app.get("/health")
def health_check():
    """Detailed health check"""
    model_loaded = predictor.model is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "features_count": len(predictor.feature_names) if predictor.feature_names else 0
    }


@app.post("/predict_burnout", response_model=BurnoutRiskResponse)
def predict_burnout(driver_pattern: DriverWorkPattern):
    """
    Predict burnout risk for a single driver based on current work patterns
    
    Returns burnout probability, risk level, and personalized recommendations
    """
    if predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train model first.")
    
    # Convert to DataFrame
    driver_data = pd.DataFrame([driver_pattern.dict()])
    driver_id = driver_data['driver_id'].iloc[0]
    
    # Remove driver_id from features
    features = driver_data.drop('driver_id', axis=1)
    
    # Predict
    burnout_prob, risk_level = predictor.predict_burnout_risk(features)
    
    # Get feature importance for this driver's risk factors
    feature_importances = pd.DataFrame({
        'feature': predictor.feature_names,
        'importance': predictor.model.feature_importances_,
        'value': features.iloc[0].values
    }).sort_values('importance', ascending=False)
    
    top_risk_factors = feature_importances.head(5).to_dict('records')
    
    # Generate recommendations
    recommendations = _generate_recommendations(driver_pattern, risk_level[0])
    
    return {
        "driver_id": int(driver_id),
        "burnout_probability": float(burnout_prob[0]),
        "risk_level": risk_level[0],
        "top_risk_factors": top_risk_factors,
        "recommendations": recommendations
    }


@app.post("/batch_predict")
def batch_predict_burnout(drivers: List[DriverWorkPattern]):
    """Predict burnout risk for multiple drivers"""
    if predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train model first.")
    
    results = []
    for driver in drivers:
        driver_data = pd.DataFrame([driver.dict()])
        features = driver_data.drop('driver_id', axis=1)
        burnout_prob, risk_level = predictor.predict_burnout_risk(features)
        
        results.append({
            "driver_id": driver.driver_id,
            "burnout_probability": float(burnout_prob[0]),
            "risk_level": risk_level[0]
        })
    
    return {"predictions": results, "total_drivers": len(results)}


@app.get("/high_risk_drivers")
def get_high_risk_drivers(threshold: float = 0.5, limit: int = 100):
    """
    Get list of drivers with burnout probability above threshold
    
    This would typically query a database in production
    """
    # For demo purposes - in production, this would query driver_health_metrics table
    return {
        "message": "In production, this endpoint queries the database for drivers with burnout_probability > threshold",
        "threshold": threshold,
        "limit": limit,
        "example_query": "SELECT * FROM driver_health_metrics WHERE burnout_probability > ? ORDER BY burnout_probability DESC LIMIT ?"
    }


@app.post("/intervention_recommendations")
def get_intervention_recommendations(driver_pattern: DriverWorkPattern) -> List[InterventionRecommendation]:
    """
    Get detailed intervention recommendations based on driver work patterns
    """
    recommendations = []
    
    # Overwork detection
    if driver_pattern.avg_weekly_hours > 50:
        recommendations.append({
            "recommendation_type": "workload_reduction",
            "priority": "high",
            "message": f"Driver is working {driver_pattern.avg_weekly_hours:.1f} hours/week, which exceeds sustainable levels",
            "action_items": [
                "Suggest reducing weekly hours to 40-45",
                "Encourage at least 1 full day off per week",
                "Monitor for next 2 weeks"
            ]
        })
    
    # Consecutive days warning
    if driver_pattern.max_consecutive_days > 7:
        recommendations.append({
            "recommendation_type": "rest_break_needed",
            "priority": "critical" if driver_pattern.max_consecutive_days > 10 else "high",
            "message": f"Driver has worked {driver_pattern.max_consecutive_days} consecutive days without break",
            "action_items": [
                "Recommend immediate 2-day rest period",
                "Send wellness check notification",
                "Consider temporary schedule restrictions"
            ]
        })
    
    # Insufficient breaks
    if driver_pattern.avg_break_duration < 15 or driver_pattern.breaks_per_shift < 1.5:
        recommendations.append({
            "recommendation_type": "break_improvement",
            "priority": "medium",
            "message": "Driver is not taking adequate breaks during shifts",
            "action_items": [
                "Send break reminder notifications every 3 hours",
                "Educate on importance of breaks for safety and earnings",
                "Highlight recommended break locations"
            ]
        })
    
    # Late night fatigue
    if driver_pattern.late_night_shifts_pct > 0.5:
        recommendations.append({
            "recommendation_type": "sleep_health",
            "priority": "medium",
            "message": f"{driver_pattern.late_night_shifts_pct*100:.0f}% of shifts are late-night (11pm-5am)",
            "action_items": [
                "Provide sleep health resources",
                "Suggest limiting late-night shifts to 2-3 per week",
                "Monitor for signs of fatigue"
            ]
        })
    
    # Declining earnings = performance degradation
    if driver_pattern.earnings_trend == -1 or driver_pattern.earnings_decline_pct < -0.15:
        recommendations.append({
            "recommendation_type": "performance_support",
            "priority": "high",
            "message": "Earnings have declined significantly, indicating possible fatigue or disengagement",
            "action_items": [
                "Offer earnings optimization training",
                "Review route and timing strategies",
                "Check for underlying health or wellbeing issues"
            ]
        })
    
    # Long shift warnings
    if driver_pattern.long_shifts_pct > 0.5:
        recommendations.append({
            "recommendation_type": "shift_duration",
            "priority": "medium",
            "message": f"{driver_pattern.long_shifts_pct*100:.0f}% of shifts exceed 10 hours",
            "action_items": [
                "Recommend shift duration of 8-10 hours max",
                "Highlight safety risks of extended driving",
                "Suggest break points for long shifts"
            ]
        })
    
    # If no specific concerns
    if len(recommendations) == 0:
        recommendations.append({
            "recommendation_type": "healthy_pattern",
            "priority": "low",
            "message": "Driver is maintaining healthy work patterns",
            "action_items": [
                "Continue monitoring",
                "Provide positive reinforcement",
                "Encourage maintaining current schedule"
            ]
        })
    
    return recommendations


def _generate_recommendations(driver_pattern: DriverWorkPattern, risk_level: str) -> List[str]:
    """Generate quick recommendation strings"""
    recs = []
    
    if risk_level in ['high', 'critical']:
        recs.append("⚠️ URGENT: Schedule wellness check with driver")
    
    if driver_pattern.max_consecutive_days > 7:
        recs.append(f"Recommend immediate rest period (worked {driver_pattern.max_consecutive_days} consecutive days)")
    
    if driver_pattern.avg_weekly_hours > 50:
        recs.append(f"Reduce weekly hours from {driver_pattern.avg_weekly_hours:.1f} to 40-45")
    
    if driver_pattern.avg_break_duration < 15:
        recs.append("Increase break duration (currently averaging only {:.0f} minutes)".format(driver_pattern.avg_break_duration))
    
    if driver_pattern.long_shifts_pct > 0.5:
        recs.append("Limit shifts to 8-10 hours maximum")
    
    if driver_pattern.late_night_shifts_pct > 0.5:
        recs.append("Reduce late-night shift frequency for better sleep health")
    
    if driver_pattern.earnings_trend == -1:
        recs.append("Declining earnings detected - offer support and training")
    
    if len(recs) == 0:
        recs.append("✓ Driver maintaining healthy work patterns - continue monitoring")
    
    return recs


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("Driver Burnout Prevention API")
    print("="*60)
    print("\nStarting server on http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
