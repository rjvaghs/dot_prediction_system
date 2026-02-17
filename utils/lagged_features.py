import pandas as pd
import numpy as np

LAGS = 120

df_year = pd.read_csv("C:/dot_prediction_system/data/flow_rate_1000000471.csv")

def prepare_training_dataset(df_year):

    # --------------------------------------------------
    # 1. Load raw irregular data
    # --------------------------------------------------
    df_raw = df_year
    df_raw = df_raw.sort_values("timestamp")

    # --------------------------------------------------
    # 2. Resample to 1-minute
    # --------------------------------------------------
    df_1min = (
        df_raw
        .set_index("timestamp")
        .resample("1min")
        .agg({
            "flow_kg": "mean",
            "current_stock": "last",
            "hour": "last",
            "minute": "last",
            "day_of_week": "last",
            "is_weekend": "last",
            "sin_hour": "last",
            "cos_hour": "last"
        })
        .dropna()
        .reset_index()
    )

    # --------------------------------------------------
    # 3. Create lag features
    # --------------------------------------------------
    for i in range(1, LAGS + 1):
        df_1min[f"flow_t-{i}"] = df_1min["flow_kg"].shift(i)

    # --------------------------------------------------
    # 4. Remove refill-crossing rows
    # --------------------------------------------------
    df_clean = (
        df_1min
        .dropna()
        .drop(columns=["current_stock"])
    )

    # --------------------------------------------------
    # 5. Save ML-ready dataset
    # --------------------------------------------------
    df_clean.to_csv("C:/dot_prediction_system/data/flow_rate_1000000518.csv", index=False)

    print("Training dataset prepared.")
    return df_clean

df = prepare_training_dataset(df_year)
df.to_csv('C:/dot_prediction_system/data/training_data_1000000471.csv')