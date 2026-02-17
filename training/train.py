import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

def train_model(STATION_ID):
    print("start")

    DATA_PATH = f"data/training_data_{STATION_ID}.csv"
    EXPERIMENT_NAME = "dot_prediction"
    MODEL_NAME = f"station_{STATION_ID}"

    import os
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ---------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------

    print("Load Data")

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")


    # --------------------------------------------------
    # DROP UNUSED COLUMNS
    # --------------------------------------------------

    TARGET = "flow_kg"
    FEATURES = [c for c in df.columns if c not in ["timestamp", TARGET]]

    # ---------------------------------------------------
    # TIME-BASED SPLIT
    # ---------------------------------------------------

    print("SPlit data")

    split_time = df["timestamp"].quantile(0.8)

    train_df = df[df["timestamp"] <= split_time]
    test_df  = df[df["timestamp"] > split_time]

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    # ---------------------------------------------------
    # TRAIN
    # ---------------------------------------------------

    print("Begin training")

    with mlflow.start_run():

        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42
        )

        print("Model fitting")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        mlflow.log_param("lags", 120)

        print("Logging model")

        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        print("Training complete.")
        print("MAE:", mae)
        print("RMSE:", rmse)

stations = ["1000000518", "1000000471", "1000000523"]

for i in stations:
    train_model(i)