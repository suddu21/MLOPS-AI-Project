import pandas as pd
import os
import logging

RAW_DATA_PATH = 'creditcard.csv'
OUTPUT_PATH = 'processed_data.csv'

logging.basicConfig(level=logging.INFO)

def ingest_data():
    df = pd.read_csv(RAW_DATA_PATH)
    required_columns = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Missing required columns in dataset")
    #os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"Ingested data saved to {OUTPUT_PATH}")

if __name__ == '__main__':
    ingest_data()