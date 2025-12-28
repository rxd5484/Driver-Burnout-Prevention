-- Driver Fatigue & Burnout Prevention Database Schema
-- Tracks driver work patterns, break behavior, and burnout risk factors

DROP TABLE IF EXISTS shift_breaks;
DROP TABLE IF EXISTS shifts;
DROP TABLE IF EXISTS driver_health_metrics;
DROP TABLE IF EXISTS drivers;

-- Core driver information
CREATE TABLE drivers (
    driver_id INT PRIMARY KEY AUTO_INCREMENT,
    driver_uuid VARCHAR(36) UNIQUE NOT NULL,
    signup_date DATE NOT NULL,
    vehicle_type VARCHAR(20), -- 'standard', 'xl', 'lux'
    preferred_shift_type VARCHAR(20), -- 'morning', 'evening', 'night', 'flexible'
    work_experience_years DECIMAL(4,2),
    average_rating DECIMAL(3,2),
    total_lifetime_trips INT DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    churn_date DATE,
    churn_reason VARCHAR(100), -- 'burnout', 'health', 'better_opportunity', 'relocation', etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_active_status (is_active),
    INDEX idx_signup_date (signup_date),
    INDEX idx_churn (churn_date, churn_reason)
);

-- Individual shift records
CREATE TABLE shifts (
    shift_id INT PRIMARY KEY AUTO_INCREMENT,
    driver_id INT NOT NULL,
    shift_date DATE NOT NULL,
    shift_start DATETIME NOT NULL,
    shift_end DATETIME NOT NULL,
    shift_duration_hours DECIMAL(5,2), -- total hours worked
    active_driving_hours DECIMAL(5,2), -- hours with passengers
    idle_hours DECIMAL(5,2), -- hours waiting for rides
    total_trips INT DEFAULT 0,
    total_earnings DECIMAL(8,2),
    earnings_per_hour DECIMAL(7,2),
    is_late_night BOOLEAN DEFAULT FALSE, -- shift includes hours between 11pm-5am
    is_weekend BOOLEAN DEFAULT FALSE,
    consecutive_days_worked INT, -- rolling count of days worked without break
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (driver_id) REFERENCES drivers(driver_id),
    INDEX idx_driver_date (driver_id, shift_date),
    INDEX idx_shift_datetime (shift_start, shift_end),
    INDEX idx_consecutive_days (consecutive_days_worked)
);

-- Break patterns during shifts
CREATE TABLE shift_breaks (
    break_id INT PRIMARY KEY AUTO_INCREMENT,
    shift_id INT NOT NULL,
    break_start DATETIME NOT NULL,
    break_end DATETIME NOT NULL,
    break_duration_minutes INT,
    break_type VARCHAR(20), -- 'meal', 'rest', 'personal'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (shift_id) REFERENCES shifts(shift_id),
    INDEX idx_shift_break (shift_id, break_start)
);

-- Aggregated health/fatigue metrics per driver (updated daily)
CREATE TABLE driver_health_metrics (
    metric_id INT PRIMARY KEY AUTO_INCREMENT,
    driver_id INT NOT NULL,
    metric_date DATE NOT NULL,
    avg_weekly_hours DECIMAL(5,2), -- rolling 7-day average
    max_consecutive_days INT, -- longest streak in past 30 days
    total_late_night_shifts_30d INT, -- count in past 30 days
    avg_break_duration_minutes DECIMAL(5,2), -- average break length
    total_breaks_per_shift DECIMAL(4,2), -- average breaks taken
    earnings_trend VARCHAR(20), -- 'increasing', 'stable', 'declining'
    fatigue_score DECIMAL(5,2), -- 0-100 composite score
    burnout_risk_level VARCHAR(20), -- 'low', 'medium', 'high', 'critical'
    burnout_probability DECIMAL(5,4), -- ML model predicted probability
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (driver_id) REFERENCES drivers(driver_id),
    UNIQUE KEY unique_driver_date (driver_id, metric_date),
    INDEX idx_driver_date_metrics (driver_id, metric_date),
    INDEX idx_burnout_risk (burnout_risk_level, burnout_probability)
);

-- View: Driver work pattern summary
CREATE VIEW driver_work_patterns AS
SELECT 
    d.driver_id,
    d.driver_uuid,
    d.signup_date,
    d.is_active,
    d.churn_date,
    d.churn_reason,
    DATEDIFF(COALESCE(d.churn_date, CURDATE()), d.signup_date) as tenure_days,
    COUNT(DISTINCT s.shift_id) as total_shifts,
    AVG(s.shift_duration_hours) as avg_shift_hours,
    AVG(s.earnings_per_hour) as avg_hourly_earnings,
    MAX(s.consecutive_days_worked) as max_consecutive_days,
    SUM(CASE WHEN s.is_late_night = TRUE THEN 1 ELSE 0 END) as total_late_night_shifts,
    AVG(CASE WHEN s.shift_duration_hours > 10 THEN 1 ELSE 0 END) as pct_long_shifts,
    (SELECT AVG(break_duration_minutes) FROM shift_breaks sb JOIN shifts s2 ON sb.shift_id = s2.shift_id WHERE s2.driver_id = d.driver_id) as avg_break_minutes
FROM drivers d
LEFT JOIN shifts s ON d.driver_id = s.driver_id
GROUP BY d.driver_id;

-- View: High burnout risk drivers (for intervention)
CREATE VIEW high_risk_drivers AS
SELECT 
    d.driver_id,
    d.driver_uuid,
    dhm.metric_date,
    dhm.avg_weekly_hours,
    dhm.max_consecutive_days,
    dhm.total_late_night_shifts_30d,
    dhm.avg_break_duration_minutes,
    dhm.fatigue_score,
    dhm.burnout_probability,
    dhm.burnout_risk_level,
    d.average_rating,
    d.total_lifetime_trips
FROM driver_health_metrics dhm
JOIN drivers d ON dhm.driver_id = d.driver_id
WHERE dhm.burnout_risk_level IN ('high', 'critical')
  AND d.is_active = TRUE
ORDER BY dhm.burnout_probability DESC;
