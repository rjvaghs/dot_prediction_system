FROM ghcr.io/mlflow/mlflow:latest

RUN pip install --no-cache-dir \
    xgboost==3.2.0 \
    pandas \
    numpy \
    scikit-learn \
    boto3
