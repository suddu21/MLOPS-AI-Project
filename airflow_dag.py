from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import logging

# Define paths (adjust these based on your environment)
RAW_DATA_PATH = '/path/to/creditcard.csv'
PROCESSED_DATA_PATH = '/path/to/processed_data.csv'
TRANSFORMED_DATA_PATH = '/path/to/transformed_data.csv'

def ingest_data():
    """Ingest raw data from CSV."""
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        # Basic validation
        required_columns = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Missing required columns in dataset")
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        logging.info("Data ingestion completed successfully")
    except Exception as e:
        logging.error(f"Data ingestion failed: {str(e)}")
        raise

def preprocess_data():
    """Preprocess data: handle missing values, scale features, and balance dataset."""
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)

        # Handle missing values
        df.fillna(df.mean(), inplace=True)

        # Separate features and target
        X = df.drop('Class', axis=1)
        y = df['Class']

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        # Balance dataset using SMOTE
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

        # Combine balanced data
        balanced_df = pd.concat([pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced, name='Class')], axis=1)

        # Save transformed data
        balanced_df.to_csv(TRANSFORMED_DATA_PATH, index=False)
        logging.info("Data preprocessing completed successfully")
    except Exception as e:
        logging.error(f"Data preprocessing failed: {str(e)}")
        raise

# Define the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'credit_card_fraud_pipeline',
    default_args=default_args,
    description='Data engineering pipeline for credit card fraud detection',
    schedule_interval='@daily',
    start_date=datetime(2025, 4, 27),
    catchup=False,
) as dag:
    ingest_task = PythonOperator(
        task_id='ingest_data',
        python_callable=ingest_data,
    )

    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
    )

    ingest_task >> preprocess_task