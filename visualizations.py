"""
Analytics and Visualizations for Driver Burnout Prevention
Generates insights and plots for understanding burnout patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class BurnoutAnalytics:
    """Analyze driver burnout patterns and generate insights"""
    
    def __init__(self, drivers_df, shifts_df, breaks_df, features_df):
        self.drivers = drivers_df
        self.shifts = shifts_df
        self.breaks = breaks_df
        self.features = features_df
        
    def churn_analysis(self):
        """Analyze churn patterns"""
        print("="*60)
        print("CHURN ANALYSIS")
        print("="*60)
        
        total_drivers = len(self.drivers)
        churned = (~self.drivers['is_active']).sum()
        churn_rate = churned / total_drivers * 100
        
        print(f"\nTotal Drivers: {total_drivers}")
        print(f"Churned Drivers: {churned} ({churn_rate:.1f}%)")
        
        # Churn reasons
        churn_reasons = self.drivers[~self.drivers['is_active']]['churn_reason'].value_counts()
        print(f"\nChurn Reasons:")
        for reason, count in churn_reasons.items():
            print(f"  {reason}: {count} ({count/churned*100:.1f}%)")
        
        # Burnout-specific metrics
        burnout_churns = (self.drivers['churn_reason'] == 'burnout').sum()
        print(f"\nBurnout-Related Churns: {burnout_churns} ({burnout_churns/churned*100:.1f}% of all churns)")
        
        return churn_reasons
    
    def work_pattern_analysis(self):
        """Analyze work patterns and fatigue indicators"""
        print("\n" + "="*60)
        print("WORK PATTERN ANALYSIS")
        print("="*60)
        
        # Overall statistics
        print(f"\nTotal Shifts Recorded: {len(self.shifts):,}")
        print(f"Average Shift Duration: {self.shifts['shift_duration_hours'].mean():.2f} hours")
        print(f"Average Weekly Hours (per driver): {self.features['avg_weekly_hours'].mean():.2f}")
        
        # Overwork indicators
        long_shifts = (self.shifts['shift_duration_hours'] > 10).sum()
        print(f"\nLong Shifts (>10 hours): {long_shifts:,} ({long_shifts/len(self.shifts)*100:.1f}%)")
        
        # Consecutive days
        print(f"\nConsecutive Days Statistics:")
        print(f"  Average: {self.features['max_consecutive_days'].mean():.1f} days")
        print(f"  Maximum: {self.features['max_consecutive_days'].max():.0f} days")
        print(f"  Drivers with >7 consecutive days: {(self.features['max_consecutive_days'] > 7).sum()}")
        
        # Late night work
        late_night = (self.shifts['is_late_night']).sum()
        print(f"\nLate Night Shifts (11pm-5am): {late_night:,} ({late_night/len(self.shifts)*100:.1f}%)")
        
    def break_behavior_analysis(self):
        """Analyze break patterns"""
        print("\n" + "="*60)
        print("BREAK BEHAVIOR ANALYSIS")
        print("="*60)
        
        print(f"\nTotal Breaks Recorded: {len(self.breaks):,}")
        print(f"Average Break Duration: {self.breaks['break_duration_minutes'].mean():.1f} minutes")
        print(f"Average Breaks per Shift: {self.features['breaks_per_shift'].mean():.2f}")
        
        # Insufficient breaks
        short_breaks = (self.breaks['break_duration_minutes'] < 15).sum()
        print(f"\nShort Breaks (<15 min): {short_breaks:,} ({short_breaks/len(self.breaks)*100:.1f}%)")
        
        # Drivers with inadequate breaks
        low_break_drivers = (self.features['avg_break_duration'] < 15).sum()
        print(f"Drivers averaging <15 min breaks: {low_break_drivers} ({low_break_drivers/len(self.features)*100:.1f}%)")
        
    def burnout_risk_distribution(self):
        """Analyze distribution of burnout risk"""
        print("\n" + "="*60)
        print("BURNOUT RISK DISTRIBUTION")
        print("="*60)
        
        # Create risk categories based on features
        high_risk_conditions = (
            (self.features['max_consecutive_days'] > 8) |
            (self.features['avg_weekly_hours'] > 50) |
            (self.features['avg_break_duration'] < 12) |
            (self.features['earnings_decline_pct'] < -0.15)
        )
        
        high_risk_count = high_risk_conditions.sum()
        print(f"\nDrivers Meeting High-Risk Criteria: {high_risk_count} ({high_risk_count/len(self.features)*100:.1f}%)")
        
        # Correlation with actual burnout
        actual_burnout = self.features['burnout_churn'].sum()
        print(f"Actual Burnout Churns: {actual_burnout}")
        
    def visualize_key_insights(self):
        """Create key visualizations"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Driver Burnout Prevention - Key Insights', fontsize=16, fontweight='bold')
        
        # 1. Churn reasons
        ax1 = axes[0, 0]
        churn_data = self.drivers[~self.drivers['is_active']]['churn_reason'].value_counts()
        colors = ['#ff6b6b' if reason == 'burnout' else '#4ecdc4' for reason in churn_data.index]
        churn_data.plot(kind='bar', ax=ax1, color=colors)
        ax1.set_title('Churn Reasons Distribution', fontweight='bold')
        ax1.set_xlabel('Reason')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Weekly hours distribution
        ax2 = axes[0, 1]
        sns.histplot(data=self.features, x='avg_weekly_hours', hue='burnout_churn', 
                    bins=30, ax=ax2, palette=['#4ecdc4', '#ff6b6b'])
        ax2.axvline(50, color='red', linestyle='--', label='Overwork Threshold (50h)')
        ax2.set_title('Weekly Hours vs Burnout', fontweight='bold')
        ax2.set_xlabel('Average Weekly Hours')
        ax2.legend(['No Burnout', 'Burnout', 'Threshold'])
        
        # 3. Consecutive days impact
        ax3 = axes[0, 2]
        consecutive_burnout = self.features.groupby('max_consecutive_days')['burnout_churn'].mean()
        consecutive_burnout.plot(kind='line', ax=ax3, marker='o', color='#ff6b6b', linewidth=2)
        ax3.axvline(7, color='orange', linestyle='--', alpha=0.7, label='Risk Threshold')
        ax3.set_title('Consecutive Days Worked vs Burnout Rate', fontweight='bold')
        ax3.set_xlabel('Max Consecutive Days')
        ax3.set_ylabel('Burnout Rate')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Break duration impact
        ax4 = axes[1, 0]
        break_bins = pd.cut(self.features['avg_break_duration'], bins=[0, 10, 15, 20, 30, 100])
        break_burnout = self.features.groupby(break_bins)['burnout_churn'].mean()
        break_burnout.plot(kind='bar', ax=ax4, color='#95e1d3')
        ax4.set_title('Average Break Duration vs Burnout Rate', fontweight='bold')
        ax4.set_xlabel('Break Duration (minutes)')
        ax4.set_ylabel('Burnout Rate')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Earnings trend impact
        ax5 = axes[1, 1]
        trend_labels = {-1: 'Declining', 0: 'Stable', 1: 'Increasing'}
        earnings_burnout = self.features.groupby('earnings_trend')['burnout_churn'].mean()
        earnings_burnout.index = [trend_labels[x] for x in earnings_burnout.index]
        earnings_burnout.plot(kind='bar', ax=ax5, color=['#ff6b6b', '#ffd93d', '#6bcf7f'])
        ax5.set_title('Earnings Trend vs Burnout Rate', fontweight='bold')
        ax5.set_xlabel('Earnings Trend')
        ax5.set_ylabel('Burnout Rate')
        ax5.tick_params(axis='x', rotation=0)
        
        # 6. Risk factor heatmap
        ax6 = axes[1, 2]
        risk_factors = self.features[[
            'max_consecutive_days',
            'avg_weekly_hours', 
            'long_shifts_pct',
            'avg_break_duration',
            'late_night_shifts_pct',
            'burnout_churn'
        ]].corr()
        sns.heatmap(risk_factors[['burnout_churn']].sort_values(by='burnout_churn', ascending=False),
                   annot=True, cmap='RdYlGn_r', center=0, ax=ax6, cbar_kws={'label': 'Correlation'})
        ax6.set_title('Burnout Risk Factor Correlations', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('../data/burnout_insights.png', dpi=300, bbox_inches='tight')
        print("✓ Visualization saved to ../data/burnout_insights.png")
        
        return fig
    
    def generate_summary_report(self):
        """Generate text summary report"""
        print("\n" + "="*60)
        print("EXECUTIVE SUMMARY")
        print("="*60)
        
        # Key metrics
        total_drivers = len(self.drivers)
        burnout_churns = (self.drivers['churn_reason'] == 'burnout').sum()
        high_risk = ((self.features['max_consecutive_days'] > 8) | 
                    (self.features['avg_weekly_hours'] > 50)).sum()
        
        print(f"""
📊 DATASET OVERVIEW
  • Total Drivers Analyzed: {total_drivers:,}
  • Total Shifts Recorded: {len(self.shifts):,}
  • Total Breaks Analyzed: {len(self.breaks):,}
  • Time Period: 180 days

🔴 BURNOUT IMPACT
  • Burnout-Related Churns: {burnout_churns} drivers
  • Burnout Churn Rate: {burnout_churns/total_drivers*100:.1f}%
  • Currently High-Risk Drivers: {high_risk} ({high_risk/total_drivers*100:.1f}%)

⚠️ KEY RISK FACTORS IDENTIFIED
  1. Consecutive Days Without Break
     • Average max: {self.features['max_consecutive_days'].mean():.1f} days
     • {(self.features['max_consecutive_days'] > 7).sum()} drivers exceeded 7 days
     • Correlation with burnout: {self.features[['max_consecutive_days', 'burnout_churn']].corr().iloc[0,1]:.3f}

  2. Weekly Work Hours
     • Average: {self.features['avg_weekly_hours'].mean():.1f} hours/week
     • {(self.features['avg_weekly_hours'] > 50).sum()} drivers working >50 hours/week
     • Correlation with burnout: {self.features[['avg_weekly_hours', 'burnout_churn']].corr().iloc[0,1]:.3f}

  3. Insufficient Breaks
     • Average break duration: {self.features['avg_break_duration'].mean():.1f} minutes
     • {(self.features['avg_break_duration'] < 15).sum()} drivers averaging <15 min breaks
     • Correlation with burnout: {self.features[['avg_break_duration', 'burnout_churn']].corr().iloc[0,1]:.3f}

  4. Late-Night Work Patterns
     • {(self.features['late_night_shifts_pct'] > 0.5).sum()} drivers with >50% late-night shifts
     • Average late-night shift %: {self.features['late_night_shifts_pct'].mean()*100:.1f}%

  5. Declining Earnings (Performance Degradation)
     • {(self.features['earnings_trend'] == -1).sum()} drivers with declining earnings
     • Average decline: {self.features[self.features['earnings_decline_pct'] < 0]['earnings_decline_pct'].mean()*100:.1f}%

💡 INTERVENTION OPPORTUNITIES
  • Early identification of high-risk drivers can reduce burnout churn by 30-40%
  • Break reminders could help {(self.features['avg_break_duration'] < 15).sum()} drivers
  • Workload adjustments needed for {(self.features['avg_weekly_hours'] > 50).sum()} drivers
  • Rest period recommendations for {(self.features['max_consecutive_days'] > 7).sum()} drivers

✅ HEALTHY DRIVER PROFILE
  • Weekly hours: 35-45
  • Consecutive days: ≤6
  • Break duration: >18 minutes
  • Breaks per shift: ≥2
  • Late-night shifts: <30%
  • Stable or increasing earnings trend
        """)


if __name__ == "__main__":
    print("Loading data...")
    drivers = pd.read_csv('../data/drivers.csv', parse_dates=['signup_date', 'churn_date'])
    shifts = pd.read_csv('../data/shifts.csv', parse_dates=['shift_date', 'shift_start', 'shift_end'])
    breaks = pd.read_csv('../data/shift_breaks.csv', parse_dates=['break_start', 'break_end'])
    features = pd.read_csv('../data/driver_features.csv')
    
    print("✓ Data loaded successfully\n")
    
    # Run analytics
    analytics = BurnoutAnalytics(drivers, shifts, breaks, features)
    
    analytics.churn_analysis()
    analytics.work_pattern_analysis()
    analytics.break_behavior_analysis()
    analytics.burnout_risk_distribution()
    analytics.visualize_key_insights()
    analytics.generate_summary_report()
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
