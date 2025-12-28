"""
Synthetic Data Generator for Driver Burnout Prevention System
Creates realistic driver work patterns with burnout risk indicators
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

class DriverDataGenerator:
    """Generate realistic driver shift patterns and burnout scenarios"""
    
    def __init__(self, n_drivers=500, days=180):
        self.n_drivers = n_drivers
        self.days = days
        self.start_date = datetime.now() - timedelta(days=days)
        
    def generate_drivers(self):
        """Generate driver profiles with varied characteristics"""
        drivers = []
        
        for i in range(self.n_drivers):
            # Varied signup dates (some new, some experienced)
            signup_offset = random.randint(0, 365 * 3)  # Up to 3 years ago
            signup_date = self.start_date - timedelta(days=signup_offset)
            
            # Some drivers will churn due to burnout
            will_churn = random.random() < 0.25  # 25% churn rate
            
            driver = {
                'driver_id': i + 1,
                'driver_uuid': str(uuid.uuid4()),
                'signup_date': signup_date.date(),
                'vehicle_type': random.choice(['standard'] * 70 + ['xl'] * 20 + ['lux'] * 10),
                'preferred_shift_type': random.choice(['morning', 'evening', 'night', 'flexible']),
                'work_experience_years': round(random.uniform(0.1, 8.0), 2),
                'average_rating': round(random.uniform(4.2, 5.0), 2),
                'total_lifetime_trips': random.randint(50, 5000),
                'is_active': not will_churn or random.random() > 0.5,  # Some churned recently
                'churn_date': None,
                'churn_reason': None
            }
            
            # If churned, set churn date and reason
            if will_churn and not driver['is_active']:
                churn_offset = random.randint(30, 150)
                driver['churn_date'] = (self.start_date + timedelta(days=churn_offset)).date()
                driver['churn_reason'] = random.choice([
                    'burnout', 'burnout', 'burnout',  # Weight towards burnout
                    'health', 'better_opportunity', 'relocation', 'family_reasons'
                ])
            
            drivers.append(driver)
        
        return pd.DataFrame(drivers)
    
    def generate_shifts(self, drivers_df):
        """Generate shift records with realistic patterns and burnout indicators"""
        shifts = []
        shift_id = 1
        
        for _, driver in drivers_df.iterrows():
            driver_id = driver['driver_id']
            signup_date = driver['signup_date']
            churn_date = driver['churn_date']
            preferred_shift = driver['preferred_shift_type']
            
            # Determine driver burnout trajectory
            burnout_trajectory = random.choice(['healthy', 'moderate_risk', 'high_risk'])
            
            # Generate shifts from signup to churn (or present)
            current_date = max(signup_date, self.start_date.date())
            end_date = churn_date if churn_date else (self.start_date + timedelta(days=self.days)).date()
            
            consecutive_days = 0
            last_shift_date = None
            
            while current_date <= end_date:
                # Probability of working depends on trajectory
                if burnout_trajectory == 'healthy':
                    work_prob = 0.60 if consecutive_days < 5 else 0.20  # Takes regular breaks
                elif burnout_trajectory == 'moderate_risk':
                    work_prob = 0.75 if consecutive_days < 7 else 0.40
                else:  # high_risk - overworking pattern
                    work_prob = 0.90 if consecutive_days < 12 else 0.50
                
                if random.random() < work_prob:
                    # Generate shift
                    shift = self._generate_shift(
                        shift_id, driver_id, current_date, preferred_shift,
                        consecutive_days, burnout_trajectory
                    )
                    shifts.append(shift)
                    shift_id += 1
                    
                    # Update consecutive days
                    if last_shift_date and (current_date - last_shift_date).days == 1:
                        consecutive_days += 1
                    else:
                        consecutive_days = 1
                    last_shift_date = current_date
                else:
                    consecutive_days = 0
                
                current_date += timedelta(days=1)
        
        return pd.DataFrame(shifts)
    
    def _generate_shift(self, shift_id, driver_id, shift_date, preferred_shift, consecutive_days, burnout_trajectory):
        """Generate individual shift with realistic timing and earnings"""
        
        # Shift timing based on preference
        if preferred_shift == 'morning':
            start_hour = random.randint(6, 9)
        elif preferred_shift == 'evening':
            start_hour = random.randint(16, 19)
        elif preferred_shift == 'night':
            start_hour = random.randint(21, 23)
        else:  # flexible
            start_hour = random.randint(6, 20)
        
        shift_start = datetime.combine(shift_date, datetime.min.time()) + timedelta(hours=start_hour)
        
        # Shift duration - longer for high burnout risk
        if burnout_trajectory == 'healthy':
            duration = random.uniform(6, 9)
        elif burnout_trajectory == 'moderate_risk':
            duration = random.uniform(8, 11)
        else:  # high_risk
            duration = random.uniform(10, 14)  # Dangerously long shifts
        
        # Fatigue penalty for consecutive days
        if consecutive_days > 7:
            duration *= random.uniform(1.1, 1.3)  # Working even longer when fatigued
        
        shift_end = shift_start + timedelta(hours=duration)
        
        # Active vs idle time
        utilization_rate = random.uniform(0.60, 0.85)
        active_hours = duration * utilization_rate
        idle_hours = duration - active_hours
        
        # Trips and earnings
        trips_per_active_hour = random.uniform(2.5, 4.5)
        total_trips = int(active_hours * trips_per_active_hour)
        
        # Earnings decline with burnout/fatigue
        base_earnings_per_hour = random.uniform(25, 40)
        if burnout_trajectory == 'high_risk' and consecutive_days > 5:
            earnings_multiplier = random.uniform(0.70, 0.90)  # Declining performance
        else:
            earnings_multiplier = random.uniform(0.95, 1.10)
        
        earnings_per_hour = base_earnings_per_hour * earnings_multiplier
        total_earnings = round(earnings_per_hour * active_hours, 2)
        
        # Late night shift flag
        is_late_night = (start_hour >= 23) or (start_hour <= 5)
        is_weekend = shift_date.weekday() >= 5
        
        return {
            'shift_id': shift_id,
            'driver_id': driver_id,
            'shift_date': shift_date,
            'shift_start': shift_start,
            'shift_end': shift_end,
            'shift_duration_hours': round(duration, 2),
            'active_driving_hours': round(active_hours, 2),
            'idle_hours': round(idle_hours, 2),
            'total_trips': total_trips,
            'total_earnings': total_earnings,
            'earnings_per_hour': round(earnings_per_hour, 2),
            'is_late_night': is_late_night,
            'is_weekend': is_weekend,
            'consecutive_days_worked': consecutive_days
        }
    
    def generate_breaks(self, shifts_df):
        """Generate break patterns during shifts"""
        breaks = []
        break_id = 1
        
        for _, shift in shifts_df.iterrows():
            shift_duration = shift['shift_duration_hours']
            shift_start = shift['shift_start']
            
            # Number of breaks based on shift length
            if shift_duration < 6:
                n_breaks = random.choice([0, 1])
            elif shift_duration < 10:
                n_breaks = random.choice([1, 2, 2])
            else:
                n_breaks = random.choice([2, 3, 3])
            
            # Drivers under stress take fewer/shorter breaks
            if shift['consecutive_days_worked'] > 8:
                n_breaks = max(0, n_breaks - random.choice([0, 1]))
            
            for i in range(n_breaks):
                # Break timing (spread throughout shift)
                break_offset_hours = random.uniform(i * shift_duration / (n_breaks + 1), 
                                                   (i + 1) * shift_duration / (n_breaks + 1))
                break_start = shift_start + timedelta(hours=break_offset_hours)
                
                # Break duration
                break_type = random.choice(['rest', 'rest', 'meal', 'personal'])
                if break_type == 'meal':
                    duration_min = random.randint(20, 45)
                elif break_type == 'personal':
                    duration_min = random.randint(10, 30)
                else:  # rest
                    duration_min = random.randint(5, 20)
                
                # Stressed drivers take shorter breaks
                if shift['consecutive_days_worked'] > 8:
                    duration_min = int(duration_min * random.uniform(0.5, 0.8))
                
                break_end = break_start + timedelta(minutes=duration_min)
                
                breaks.append({
                    'break_id': break_id,
                    'shift_id': shift['shift_id'],
                    'break_start': break_start,
                    'break_end': break_end,
                    'break_duration_minutes': duration_min,
                    'break_type': break_type
                })
                break_id += 1
        
        return pd.DataFrame(breaks)
    
    def generate_all_data(self):
        """Generate complete dataset"""
        print("Generating drivers...")
        drivers_df = self.generate_drivers()
        
        print(f"Generating shifts for {len(drivers_df)} drivers...")
        shifts_df = self.generate_shifts(drivers_df)
        
        print(f"Generating breaks for {len(shifts_df)} shifts...")
        breaks_df = self.generate_breaks(shifts_df)
        
        print(f"\nGenerated:")
        print(f"  - {len(drivers_df)} drivers")
        print(f"  - {len(shifts_df)} shifts")
        print(f"  - {len(breaks_df)} breaks")
        
        return drivers_df, shifts_df, breaks_df


if __name__ == "__main__":
    generator = DriverDataGenerator(n_drivers=500, days=180)
    drivers, shifts, breaks = generator.generate_all_data()
    
    # Save to CSV
    drivers.to_csv('../data/drivers.csv', index=False)
    shifts.to_csv('../data/shifts.csv', index=False)
    breaks.to_csv('../data/shift_breaks.csv', index=False)
    
    print("\nData saved to ../data/ directory")
    print(f"\nChurn rate: {(~drivers['is_active']).sum() / len(drivers) * 100:.1f}%")
    print(f"Burnout-related churns: {(drivers['churn_reason'] == 'burnout').sum()}")
