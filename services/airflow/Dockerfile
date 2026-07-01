FROM apache/airflow:2.9.1

USER root
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

USER airflow
RUN pip install --no-cache-dir \
    scikit-learn==1.5.0 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    mlflow==2.13.0
