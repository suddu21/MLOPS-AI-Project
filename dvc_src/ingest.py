import logging.config
import pandas as pd
import os
import logging

RAW_DATA_PATH = r'C:\Users\sudha\Desktop\My-Data\Education\IITM\2nd-sem\ML-Ops-Lab\MLOPS-AI-Project\dvc_src\data\creditcard.csv'
FEEDBACK_DATA_PATH = r'C:\Users\sudha\Desktop\My-Data\Education\IITM\2nd-sem\ML-Ops-Lab\MLOPS-AI-Project\dvc_src\data\transformed_feedback.csv'
OUTPUT_PATH = r'C:\Users\sudha\Desktop\My-Data\Education\IITM\2nd-sem\ML-Ops-Lab\MLOPS-AI-Project\dvc_src\data\processed_data.csv'

logging.basicConfig(level=logging.INFO,
                    handlers=[logging.FileHandler(r"C:\Users\sudha\Desktop\My-Data\Education\IITM\2nd-sem\ML-Ops-Lab\MLOPS-AI-Project\dvc_src\logs\ingest.log"), logging.StreamHandler()]
)

def ingest_data():
    # Read the raw data and feedback data
    raw_df = pd.read_csv(RAW_DATA_PATH)
    feedback_df = pd.read_csv(FEEDBACK_DATA_PATH)

    # Verify required columns in both datasets
    required_columns = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]
    for col in required_columns:
        if col not in raw_df.columns or col not in feedback_df.columns:
            raise ValueError(f"Missing required column '{col}' in one of the datasets")

    # Check for duplicates between raw and feedback data
    # Using all columns for deduplication to ensure exact matches
    combined_df = pd.concat([raw_df, feedback_df]).drop_duplicates(keep='first')

    # Save the combined data to the output path
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    combined_df.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"Ingested data with feedback saved to {OUTPUT_PATH}")
    logging.info(f"Total rows after deduplication: {len(combined_df)}")

if __name__ == '__main__':
    ingest_data()