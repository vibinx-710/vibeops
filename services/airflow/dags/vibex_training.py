from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
import urllib.request
import os
import logging
log = logging.getLogger(__name__)
default_args = {
    'owner': 'vibeops',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def download_data(**context):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
    urllib.request.urlretrieve(url, "/tmp/ai4i2020.csv")
    print("Dataset downloaded successfully")

def preprocess_data(**context):
    df = pd.read_csv("/tmp/ai4i2020.csv")
    
    features = ['Air temperature [K]', 'Process temperature [K]', 
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    target = 'Machine failure'
    
    X = df[features].values
    y = df[target].values
    
    np.save("/tmp/X.npy", X)
    np.save("/tmp/y.npy", y)
    print(f"Preprocessed {len(df)} rows")

def train_model(**context):
    X = np.load("/tmp/X.npy")
    y = np.load("/tmp/y.npy")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_registry_uri("http://mlflow:5000")
    os.environ['MLFLOW_TRACKING_URI'] = 'http://mlflow:5000'
    os.environ['MLFLOW_ARTIFACT_URI'] = 'mlflow-artifacts://mlflow:5000'
    mlflow.set_experiment("vibex-training")
    
    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        mlflow.sklearn.log_model(
            model,
            artifact_path="Vibex",
            registered_model_name="Vibex"
        )
        
        log.info(f"Training on {len(X)} samples")
        log.info(f"Accuracy: {accuracy:.4f}")
        log.info(f"F1 Score: {f1:.4f}")

with DAG(
    'vibex_training_pipeline',
    default_args=default_args,
    description='Vibex ML Training Pipeline',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    download = PythonOperator(
        task_id='download_data',
        python_callable=download_data,
    )

    preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
    )

    train = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )

    download >> preprocess >> train
