import os
import pandas as pd
from typing import List
from math import sin, cos, pi
from datetime import datetime

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

LAGS = 120

# ---------------------------------------------------
# FEATURE BUILDER
# ---------------------------------------------------

def build_features(timestamp: str, flow_history: List[float]):

    if len(flow_history) != LAGS:
        raise ValueError(f"flow_history must contain {LAGS} values.")

    ts = pd.Timestamp(timestamp)

    hour = ts.hour
    dow = ts.weekday()
    weekend = int(dow >= 5)

    feature_dict = {
        "hour": hour,
        "day_of_week": dow,
        "is_weekend": weekend,
        "sin_hour": sin(2 * pi * hour / 24),
        "cos_hour": cos(2 * pi * hour / 24)
    }

    # Add lag features (latest value = t-1)
    for i in range(1, LAGS + 1):
        feature_dict[f"flow_t-{i}"] = flow_history[-i]

    return pd.DataFrame([feature_dict])

import math


def build_scoring_payload(
    features: list

    recent_flow_history: list
):

    if len(recent_flow_history) != 120:
        raise ValueError("recent_flow_history must contain exactly 120 values")


    sin_hour = math.sin(2 * math.pi * hour / 24)
    cos_hour = math.cos(2 * math.pi * hour / 24)

    # ----------------------------
    # Column Order (CRITICAL)
    # ----------------------------
    columns = [
        "hour",
        "day_of_week",
        "is_weekend",
        "sin_hour",
        "cos_hour",
    ]

    # Add lag columns
    for i in range(1, 121):
        columns.append(f"flow_t-{i}")

    # Minute at the END (as per your trained model)
    columns.append("minute")

    # ----------------------------
    # Row Construction
    # ----------------------------
    row = [
        hour,
        day_of_week,
        is_weekend,
        round(sin_hour, 4),
        round(cos_hour, 4),
    ]

    # Ensure correct order flow_t-1 ... flow_t-120
    row.extend(recent_flow_history)

    # minute must be last
    row.append(minute)

    payload = {
        "dataframe_split": {
            "columns": columns,
            "data": [row]
        }
    }

    return payload
