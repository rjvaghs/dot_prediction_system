import numpy as np
import pandas as pd
from math import sin, cos, pi
from datetime import timedelta

np.random.seed(42)

def base_flow(hour):
    if 6 <= hour <= 10:
        return 12
    elif 17 <= hour <= 22:
        return 15
    elif 0 <= hour <= 5:
        return 3
    else:
        return 7

def generate_irregular_station_data(
    station_id="DBS_01",
    start_date="2025-01-01",
    days=365,
    initial_stock=6000,
    refill_threshold=800,
    refill_amount=5000
):
    end_time = pd.Timestamp(start_date) + pd.Timedelta(days=days)
    current_time = pd.Timestamp(start_date)
    stock = initial_stock
    rows = []

    while current_time < end_time:
        # Random interval between 1 and 60 seconds
        step_seconds = np.random.randint(1, 61)
        current_time += timedelta(seconds=step_seconds)

        hour = current_time.hour
        minute = current_time.minute
        dow = current_time.weekday()
        weekend = int(dow >= 5)

        base = base_flow(hour)

        # Weekend effect
        if weekend:
            base *= 0.8

        # Noise
        noise = np.random.normal(0, 0.7)

        # Rare spikes
        if np.random.rand() < 0.004:
            noise += np.random.uniform(4, 9)

        flow = max(0.3, base + noise)

        # Adjust consumption for irregular interval
        consumed = flow * (step_seconds / 60)
        stock -= consumed

        is_refill = 0
        if stock < refill_threshold:
            stock += refill_amount
            is_refill = 1

        rows.append({
            "timestamp": current_time,
            "station_id": station_id,
            "flow_kg_min": round(flow, 2),
            "consumed_kg": round(consumed, 3),
            "stock_kg": round(stock, 2),
            "is_refill": is_refill,
            "interval_sec": step_seconds,
            "hour": hour,
            "minute": minute,
            "day_of_week": dow,
            "is_weekend": weekend,
            "sin_hour": sin(2 * pi * hour / 24),
            "cos_hour": cos(2 * pi * hour / 24)
        })

    df = pd.DataFrame(rows)
    df.to_csv("data/sales.csv", header=True)

    return "DONE!!!"
