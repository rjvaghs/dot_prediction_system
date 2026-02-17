import pandas as pd
import numpy as np
from datetime import timedelta
from math import sin, cos, pi

np.random.seed(42)

# ============================================================
# LOAD REAL ONE-MONTH DAILY DATA
# ============================================================

REAL_DATA_PATH = "C:/dot_prediction_system/data/salesdata_1000000471.csv"

real_df = pd.read_csv(REAL_DATA_PATH, header=1, thousands=",")

real_df.columns = [c.strip().upper() for c in real_df.columns]

real_df["ZDATE"] = pd.to_datetime(real_df["ZDATE"])
real_df["DAY_OF_WEEK"] = real_df["ZDATE"].dt.weekday
real_df["IS_WEEKEND"] = (real_df["DAY_OF_WEEK"] >= 5).astype(int)

real_df["ZTOTAL"] = pd.to_numeric(real_df["ZTOTAL"], errors="coerce")
real_df = real_df.dropna(subset=["ZTOTAL"])

# ============================================================
# LEARN REAL DAILY PATTERNS
# ============================================================

mean_daily = real_df["ZTOTAL"].mean()
std_daily = real_df["ZTOTAL"].std()

weekday_mean = real_df[real_df["IS_WEEKEND"] == 0]["ZTOTAL"].mean()
weekend_mean = real_df[real_df["IS_WEEKEND"] == 1]["ZTOTAL"].mean()

weekend_multiplier = weekend_mean / weekday_mean if weekday_mean > 0 else 0.9

print("Mean Daily:", mean_daily)
print("Std Daily:", std_daily)
print("Weekend Multiplier:", weekend_multiplier)

# ============================================================
# HOURLY PROFILE
# ============================================================

def hourly_weight(hour):
    if 0 <= hour <= 5:
        return 0.025
    elif 6 <= hour <= 10:
        return 0.065
    elif 11 <= hour <= 16:
        return 0.045
    elif 17 <= hour <= 22:
        return 0.075
    else:
        return 0.035

hourly_weights = np.array([hourly_weight(h) for h in range(24)])
hourly_weights = hourly_weights / hourly_weights.sum()

# ============================================================
# GENERATE 1 YEAR MINUTE DATA WITH STOCK
# ============================================================

def generate_one_year_minute_data(
    start_date="2025-01-01",
    initial_stock=6000
):

    start = pd.Timestamp(start_date)
    end = start + pd.Timedelta(days=365)

    rows = []
    current_day = start
    current_stock = initial_stock

    while current_day < end:

        dow = current_day.weekday()
        weekend = dow >= 5

        # Sample daily demand
        daily_demand = np.random.normal(mean_daily, std_daily * 0.4)

        if weekend:
            daily_demand *= weekend_multiplier

        daily_demand = max(daily_demand, mean_daily * 0.6)

        hourly_demand = daily_demand * hourly_weights

        for hour in range(24):

            base_per_min = hourly_demand[hour] / 60

            for minute in range(60):

                timestamp = current_day + timedelta(hours=hour, minutes=minute)

                minute_wave = 0.08 * base_per_min * np.sin(2 * np.pi * minute / 60)
                noise = np.random.normal(0, base_per_min * 0.12)

                # rare spike
                if np.random.rand() < 0.002:
                    noise += np.random.uniform(1.5, 3.5)

                flow = max(0, base_per_min + minute_wave + noise)

                # Deduct from stock
                current_stock -= flow
                current_stock = max(current_stock, 0)

                rows.append({
                    "timestamp": timestamp,
                    "flow_kg": round(flow, 4),
                    "current_stock": round(current_stock, 2),
                    "hour": hour,
                    "minute": minute,
                    "day_of_week": dow,
                    "is_weekend": int(weekend),
                    "sin_hour": sin(2 * pi * hour / 24),
                    "cos_hour": cos(2 * pi * hour / 24)
                })

        current_day += timedelta(days=1)

    df = pd.DataFrame(rows)

    df.to_csv("C:/dot_prediction_system/data/flow_rate_1000000471.csv", index=False)

    return df

# ============================================================
# RUN
# ============================================================

df_year = generate_one_year_minute_data()

print("Generated rows:", len(df_year))
