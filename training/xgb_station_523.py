import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_absolute_error

# =====================================
# SET MLFLOW EXPERIMENT
# =====================================

mlflow.set_experiment("dot_prediction")

with mlflow.start_run():

    # =====================================
    # 1. LOAD RAW EVENT DATA
    # =====================================

    df = pd.read_csv("swaminarayan_65_sales.csv")

    df["timestamp"] = pd.to_datetime(df["Time"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print("Raw rows:", len(df))
    mlflow.log_metric("raw_rows", len(df))


    # =====================================
    # 2. CONVERT CUMULATIVE → EVENT SALES
    # =====================================

    df["event_sale"] = df["Value"].diff()
    df["event_sale"] = df["event_sale"].fillna(0)

    # Handle refill reset
    df.loc[df["event_sale"] < 0, "event_sale"] = 0


    # =====================================
    # 3. CREATE 15-MIN BUCKETS
    # =====================================

    df["bucket_start"] = df["timestamp"].dt.floor("15min")

    bucket_df = (
        df.groupby("bucket_start")["event_sale"]
        .sum()
        .reset_index()
        .rename(columns={"event_sale": "sales_15min"})
    )

    # Cap extreme outliers
    upper_cap = bucket_df["sales_15min"].quantile(0.995)
    bucket_df["sales_15min"] = np.minimum(bucket_df["sales_15min"], upper_cap)

    # Fill missing buckets
    full_range = pd.date_range(
        start=bucket_df["bucket_start"].min(),
        end=bucket_df["bucket_start"].max(),
        freq="15min"
    )

    bucket_df = bucket_df.set_index("bucket_start").reindex(full_range).fillna(0)
    bucket_df.index.name = "bucket_start"
    bucket_df = bucket_df.reset_index()

    print("15-min buckets:", len(bucket_df))
    print("Zero buckets:", (bucket_df["sales_15min"] == 0).sum())

    mlflow.log_metric("total_buckets", len(bucket_df))
    mlflow.log_metric("zero_bucket_pct",
                      (bucket_df["sales_15min"] == 0).mean() * 100)


    # =====================================
    # 4. FEATURE ENGINEERING
    # =====================================

    bucket_df = bucket_df.sort_values("bucket_start").reset_index(drop=True)

    bucket_df["hour"] = bucket_df["bucket_start"].dt.hour
    bucket_df["day_of_week"] = bucket_df["bucket_start"].dt.dayofweek

    lag_list = [1,2,3,4,8,12,24,96]

    for lag in lag_list:
        bucket_df[f"lag_{lag}"] = bucket_df["sales_15min"].shift(lag)

    bucket_df["rolling_mean_4"] = bucket_df["sales_15min"].shift(1).rolling(4).mean()
    bucket_df["rolling_mean_8"] = bucket_df["sales_15min"].shift(1).rolling(8).mean()
    bucket_df["rolling_mean_96"] = bucket_df["sales_15min"].shift(1).rolling(96).mean()

    bucket_df["target"] = bucket_df["sales_15min"].shift(-1)

    train_df = bucket_df.dropna().reset_index(drop=True)

    print("Final usable rows:", len(train_df))
    mlflow.log_metric("usable_rows", len(train_df))


    # =====================================
    # 5. TRAIN / TEST SPLIT
    # =====================================

    split_index = int(len(train_df) * 0.8)

    train_data = train_df.iloc[:split_index]
    test_data  = train_df.iloc[split_index:]

    FEATURE_COLS = [
        'lag_1','lag_2','lag_3','lag_4',
        'lag_8','lag_12','lag_24','lag_96',
        'rolling_mean_4','rolling_mean_8','rolling_mean_96',
        'hour','day_of_week'
    ]

    X_train = train_data[FEATURE_COLS]
    y_train = train_data["target"]

    X_test = test_data[FEATURE_COLS]
    y_test = test_data["target"]


    # =====================================
    # 6. TRAIN XGBOOST
    # =====================================

    params = {
        "objective": "reg:tweedie",
        "tweedie_variance_power": 1.2,
        "eval_metric": "rmse",
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.7,
        "min_child_weight": 3,
        "gamma": 0.1,
        "seed": 42
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest  = xgb.DMatrix(X_test, label=y_test)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1200,
        evals=[(dtrain, "train"), (dtest, "valid")],
        early_stopping_rounds=50,
        verbose_eval=100
    )


    # =====================================
    # 7. EVALUATION
    # =====================================

    y_pred = model.predict(dtest)

    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    mae = mean_absolute_error(y_test, y_pred)
    wape = np.sum(np.abs(y_test - y_pred)) / np.sum(np.abs(y_test)) * 100

    print("\n==== FINAL METRICS ====")
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("WAPE:", wape)

    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("wape", wape)

    # Log parameters
    mlflow.log_params(params)

    # Log feature list as artifact
    with open("feature_cols.txt", "w") as f:
        for col in FEATURE_COLS:
            f.write(col + "\n")

    mlflow.log_artifact("feature_cols.txt")

    # Log model
    mlflow.xgboost.log_model(
        model,
        artifact_path="model",
        registered_model_name="xgb_station_523"
    )

    print("Model saved and logged to MLflow.")