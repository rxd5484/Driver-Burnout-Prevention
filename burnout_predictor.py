"""
Driver Burnout Prediction Model
Uses XGBoost to predict driver churn risk based on work patterns and fatigue indicators
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import json
from datetime import datetime, timedelta

class BurnoutPredictor:
    """ML model to predict driver burnout risk"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def engineer_features(self, drivers_df, shifts_df, breaks_df):
        """
        Engineer features from raw data similar to employee attrition project
        Focus on work pattern indicators and fatigue signals
        """
        print("Engineering features from driver behavior patterns...")
        
        # Calculate driver-level aggregations from shifts
        driver_features = []
        
        for driver_id in drivers_df['driver_id'].unique():
            driver_shifts = shifts_df[shifts_df['driver_id'] == driver_id].copy()
            driver_breaks = breaks_df[breaks_df['shift_id'].isin(driver_shifts['shift_id'])]
            driver_info = drivers_df[drivers_df['driver_id'] == driver_id].iloc[0]
            
            if len(driver_shifts) == 0:
                continue
            
            # Sort by date
            driver_shifts = driver_shifts.sort_values('shift_date')
            
            # Basic driver info
            signup_date = driver_info['signup_date']
            if isinstance(signup_date, pd.Timestamp):
                signup_date = signup_date.date()
            tenure_days = (datetime.now().date() - signup_date).days
            
            # Work intensity features
            avg_weekly_hours = driver_shifts['shift_duration_hours'].sum() / (tenure_days / 7) if tenure_days > 0 else 0
            avg_shift_duration = driver_shifts['shift_duration_hours'].mean()
            max_shift_duration = driver_shifts['shift_duration_hours'].max()
            total_shifts = len(driver_shifts)
            shifts_per_week = total_shifts / (tenure_days / 7) if tenure_days > 0 else 0
            
            # Overwork indicators
            long_shifts_pct = (driver_shifts['shift_duration_hours'] > 10).mean()
            max_consecutive_days = driver_shifts['consecutive_days_worked'].max()
            avg_consecutive_days = driver_shifts['consecutive_days_worked'].mean()
            
            # Late night work (burnout risk factor)
            late_night_shifts_pct = driver_shifts['is_late_night'].mean()
            total_late_night_shifts = driver_shifts['is_late_night'].sum()
            
            # Weekend work patterns
            weekend_shifts_pct = driver_shifts['is_weekend'].mean()
            
            # Performance/earnings trends
            avg_earnings_per_hour = driver_shifts['earnings_per_hour'].mean()
            earnings_std = driver_shifts['earnings_per_hour'].std()
            earnings_trend = self._calculate_trend(driver_shifts['earnings_per_hour'].values)
            
            # Recent vs historical earnings (declining = warning sign)
            if len(driver_shifts) >= 20:
                recent_earnings = driver_shifts.tail(10)['earnings_per_hour'].mean()
                historical_earnings = driver_shifts.head(10)['earnings_per_hour'].mean()
                earnings_decline_pct = (recent_earnings - historical_earnings) / historical_earnings if historical_earnings > 0 else 0
            else:
                earnings_decline_pct = 0
            
            # Break patterns (critical for burnout prevention)
            if len(driver_breaks) > 0:
                avg_break_duration = driver_breaks['break_duration_minutes'].mean()
                breaks_per_shift = len(driver_breaks) / len(driver_shifts)
                short_break_pct = (driver_breaks['break_duration_minutes'] < 15).mean()
            else:
                avg_break_duration = 0
                breaks_per_shift = 0
                short_break_pct = 1.0
            
            # Workload variability (inconsistent schedule = stress)
            shift_duration_variability = driver_shifts['shift_duration_hours'].std()
            
            # Recent activity (last 30 days)
            recent_date = pd.Timestamp(datetime.now().date() - timedelta(days=30))
            recent_shifts = driver_shifts[driver_shifts['shift_date'] >= recent_date]
            shifts_last_30d = len(recent_shifts)
            hours_last_30d = recent_shifts['shift_duration_hours'].sum() if len(recent_shifts) > 0 else 0
            
            # Rating (deteriorating rating = burnout sign)
            current_rating = driver_info['average_rating']
            
            # Target variable: churned due to burnout
            is_burnout_churn = (driver_info['churn_reason'] == 'burnout') if pd.notna(driver_info['churn_reason']) else False
            
            driver_features.append({
                'driver_id': driver_id,
                'tenure_days': tenure_days,
                'total_shifts': total_shifts,
                'shifts_per_week': shifts_per_week,
                'avg_weekly_hours': avg_weekly_hours,
                'avg_shift_duration': avg_shift_duration,
                'max_shift_duration': max_shift_duration,
                'long_shifts_pct': long_shifts_pct,
                'max_consecutive_days': max_consecutive_days,
                'avg_consecutive_days': avg_consecutive_days,
                'late_night_shifts_pct': late_night_shifts_pct,
                'total_late_night_shifts': total_late_night_shifts,
                'weekend_shifts_pct': weekend_shifts_pct,
                'avg_earnings_per_hour': avg_earnings_per_hour,
                'earnings_std': earnings_std,
                'earnings_trend': earnings_trend,
                'earnings_decline_pct': earnings_decline_pct,
                'avg_break_duration': avg_break_duration,
                'breaks_per_shift': breaks_per_shift,
                'short_break_pct': short_break_pct,
                'shift_duration_variability': shift_duration_variability,
                'shifts_last_30d': shifts_last_30d,
                'hours_last_30d': hours_last_30d,
                'current_rating': current_rating,
                'work_experience_years': driver_info['work_experience_years'],
                'burnout_churn': int(is_burnout_churn)
            })
        
        features_df = pd.DataFrame(driver_features)
        print(f"Engineered {len(features_df.columns) - 2} features for {len(features_df)} drivers")
        
        return features_df
    
    def _calculate_trend(self, values):
        """Calculate trend direction (-1: declining, 0: stable, 1: increasing)"""
        if len(values) < 2:
            return 0
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.5:
            return 1
        elif slope < -0.5:
            return -1
        else:
            return 0
    
    def train(self, features_df, test_size=0.2):
        """Train XGBoost model with class imbalance handling"""
        print("\nTraining burnout prediction model...")
        
        # Prepare features and target
        X = features_df.drop(['driver_id', 'burnout_churn'], axis=1)
        y = features_df['burnout_churn']
        
        self.feature_names = X.columns.tolist()
        
        print(f"Dataset: {len(X)} drivers")
        print(f"Burnout rate: {y.mean()*100:.1f}%")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Handle class imbalance with SMOTE
        print("\nApplying SMOTE to balance classes...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"After SMOTE: {len(X_train_balanced)} samples")
        print(f"Balanced class distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost
        print("\nTraining XGBoost classifier...")
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='auc'
        )
        
        self.model.fit(
            X_train_scaled, y_train_balanced,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print("\n" + "="*60)
        print("MODEL PERFORMANCE")
        print("="*60)
        print(classification_report(y_test, y_pred, target_names=['No Burnout', 'Burnout']))
        print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n" + "="*60)
        print("TOP 10 BURNOUT INDICATORS")
        print("="*60)
        print(feature_importance.head(10).to_string(index=False))
        
        return {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'feature_importance': feature_importance.to_dict('records')
        }
    
    def predict_burnout_risk(self, driver_features):
        """Predict burnout probability for a driver"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Ensure feature order matches training
        X = driver_features[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        # Predict probability
        burnout_prob = self.model.predict_proba(X_scaled)[:, 1]
        
        # Risk categorization
        risk_level = []
        for prob in burnout_prob:
            if prob < 0.25:
                risk_level.append('low')
            elif prob < 0.50:
                risk_level.append('medium')
            elif prob < 0.75:
                risk_level.append('high')
            else:
                risk_level.append('critical')
        
        return burnout_prob, risk_level
    
    def save_model(self, model_path='../models/'):
        """Save trained model and scaler"""
        joblib.dump(self.model, f'{model_path}burnout_model.pkl')
        joblib.dump(self.scaler, f'{model_path}scaler.pkl')
        
        with open(f'{model_path}feature_names.json', 'w') as f:
            json.dump(self.feature_names, f)
        
        print(f"\nModel saved to {model_path}")
    
    def load_model(self, model_path='../models/'):
        """Load trained model and scaler"""
        self.model = joblib.load(f'{model_path}burnout_model.pkl')
        self.scaler = joblib.load(f'{model_path}scaler.pkl')
        
        with open(f'{model_path}feature_names.json', 'r') as f:
            self.feature_names = json.load(f)
        
        print(f"Model loaded from {model_path}")


if __name__ == "__main__":
    # Load generated data
    print("Loading data...")
    drivers_df = pd.read_csv('../data/drivers.csv', parse_dates=['signup_date', 'churn_date'])
    shifts_df = pd.read_csv('../data/shifts.csv', parse_dates=['shift_date', 'shift_start', 'shift_end'])
    breaks_df = pd.read_csv('../data/shift_breaks.csv', parse_dates=['break_start', 'break_end'])
    
    # Initialize predictor
    predictor = BurnoutPredictor()
    
    # Engineer features
    features_df = predictor.engineer_features(drivers_df, shifts_df, breaks_df)
    
    # Train model
    results = predictor.train(features_df)
    
    # Save model
    import os
    os.makedirs('../models', exist_ok=True)
    predictor.save_model()
    
    # Save feature dataset
    features_df.to_csv('../data/driver_features.csv', index=False)
    print("\nFeature dataset saved to ../data/driver_features.csv")
