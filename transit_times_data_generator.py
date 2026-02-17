import numpy as np
import pandas as pd
from datetime import timedelta
import random

np.random.seed(42)
random.seed(42)

# -----------------------
# Helper functions
# -----------------------

def is_peak(hour):
    return (7 <= hour <= 10) or (17 <= hour <= 21)

def generate_construction_windows(
    start_date, days, num_events=3, min_duration=7, max_duration=21
):
    events = []
    for _ in range(num_events):
        start = start_date + timedelta(days=random.randint(0, days - max_duration))
        duration = random.randint(min_duration, max_duration)
        end = start + timedelta(days=duration)
        events.append((start, end))
    return events

def is_under_construction(ts, construction_windows):
    return any(start <= ts <= end for start, end in construction_windows)

def sample_trips_per_day():
    trips = [0, 1, 2, 3, 4, 5, 6]
    probs = [0.003, 0.05, 0.15, 0.30, 0.25, 0.20, 0.047]
    return np.random.choice(trips, p=probs)

# -----------------------
# Main generator
# -----------------------

def generate_transit_data(
    route_id="MS01_DBS01",
    start_date="2025-01-01",
    days=365,
    base_transit_min=42,
    holidays=None
):
    start_date = pd.Timestamp(start_date)
    end_date = start_date + pd.Timedelta(days=days)

    construction_windows = generate_construction_windows(start_date, days)
    rows = []
    trip_id = 1

    current_date = start_date

    while current_date < end_date:
        trips_today = sample_trips_per_day()

        for _ in range(trips_today):
            hour = random.randint(5, 23)
            start_time = current_date + timedelta(hours=hour)

            dow = start_time.weekday()
            weekend = int(dow >= 5)
            peak = int(is_peak(hour))
            holiday = int(holidays and start_time.date() in holidays)
            construction = int(is_under_construction(start_time, construction_windows))

            transit = base_transit_min

            # Weekend effect
            if weekend:
                transit *= 0.9

            # Peak hour congestion
            if peak:
                transit += random.uniform(8, 18)

            # Holiday effect
            if holiday:
                transit += random.uniform(5, 15)

            # Construction impact
            if construction:
                transit += random.uniform(15, 30)

            # Random noise
            transit += np.random.normal(0, 3)

            # Rare extreme events
            if random.random() < 0.01:
                transit += random.uniform(20, 45)

            transit = max(20, round(transit, 1))

            rows.append({
                "trip_id": f"T{trip_id}",
                "route_id": route_id,
                "start_time": start_time,
                "day_of_week": dow,
                "hour_of_day": hour,
                "is_weekend": weekend,
                "is_peak_hour": peak,
                "is_holiday": holiday,
                "construction_active": construction,
                "base_time_min": base_transit_min,
                "transit_time_min": transit
            })

            trip_id += 1

        current_date += timedelta(days=1)

    return pd.DataFrame(rows)

import pandas as pd

holidays = {
    pd.Timestamp("2025-01-01").date(),  # New Year’s Day
    pd.Timestamp("2025-01-14").date(),  # Uttarayan / Makar Sankranti
    pd.Timestamp("2025-02-19").date(),  # Chhatrapati Shivaji Maharaj Jayanti
    pd.Timestamp("2025-03-29").date(),  # Gudi Padwa
    pd.Timestamp("2025-03-30").date(),  # Ugadi
    pd.Timestamp("2025-03-31").date(),  # Eid-ul-Fitr (Tentative)
    pd.Timestamp("2025-04-06").date(),  # Ram Navami
    pd.Timestamp("2025-04-10").date(),  # Mahavir Jayanti
    pd.Timestamp("2025-04-14").date(),  # Dr. B.R. Ambedkar Jayanti
    pd.Timestamp("2025-04-18").date(),  # Good Friday
    pd.Timestamp("2025-05-01").date(),  # Gujarat Day
    pd.Timestamp("2025-05-12").date(),  # Buddha Purnima
    pd.Timestamp("2025-05-29").date(),  # Maharana Pratap Jayanti
    pd.Timestamp("2025-06-07").date(),  # Bakrid / Eid-ul-Adha (Tentative)
    pd.Timestamp("2025-08-15").date(),  # Independence Day
    pd.Timestamp("2025-08-16").date(),  # Parsi New Year
    pd.Timestamp("2025-08-16").date(),  # Janmashtami
    pd.Timestamp("2025-08-27").date(),  # Ganesh Chaturthi
    pd.Timestamp("2025-10-02").date(),  # Gandhi Jayanti
    pd.Timestamp("2025-10-02").date(),  # Vijaya Dashami
    pd.Timestamp("2025-10-21").date(),  # Diwali
    pd.Timestamp("2025-10-22").date(),  # Gujarati New Year
    pd.Timestamp("2025-10-23").date(),  # Bhai Dooj
    pd.Timestamp("2025-10-31").date(),  # Sardar Vallabhbhai Patel Jayanti
    pd.Timestamp("2025-11-05").date(),  # Guru Nanak Jayanti
    pd.Timestamp("2025-12-25").date(),  # Christmas Day
}

df_transit = generate_transit_data(
    route_id="MS01_DBS01",
    start_date="2025-01-01",
    days=365,
    base_transit_min=42,
    holidays=holidays
)