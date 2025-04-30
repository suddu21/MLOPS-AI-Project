import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import logging

INPUT_PATH = r'C:\Users\sudha\Desktop\My-Data\Education\IITM\2nd-sem\ML-Ops-Lab\MLOPS-AI-Project\dvc_src\data\processed_data.csv'
TRANSFORMED_DATA_PATH = r'C:\Users\sudha\Desktop\My-Data\Education\IITM\2nd-sem\ML-Ops-Lab\MLOPS-AI-Project\dvc_src\data\transformed_data.csv'
TEST_DATA_PATH = r'C:\Users\sudha\Desktop\My-Data\Education\IITM\2nd-sem\ML-Ops-Lab\MLOPS-AI-Project\dvc_src\data\test_data.csv'

logging.basicConfig(level=logging.INFO,
                    handlers=[logging.FileHandler(r"C:\Users\sudha\Desktop\My-Data\Education\IITM\2nd-sem\ML-Ops-Lab\MLOPS-AI-Project\dvc_src\logs\preprocess.log"), logging.StreamHandler()]
)
logging.info("Starting data preprocessing...")

def preprocess_data():
    df = pd.read_csv(INPUT_PATH)

    # Stratified test set
    n_class_1 = min(len(df[df['Class'] == 1]), 100)
    n_class_0 = min(len(df[df['Class'] == 0]), 1000 - n_class_1)
    test_df_class_1 = df[df['Class'] == 1].sample(n=n_class_1, random_state=42)
    test_df_class_0 = df[df['Class'] == 0].sample(n=n_class_0, random_state=42)
    test_df = pd.concat([test_df_class_1, test_df_class_0]).sample(frac=1, random_state=42)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    logging.info(f"Test data saved to {TEST_DATA_PATH}")
    logging.info(f"Test set size: {len(test_df)}")

    # Remove test set
    df = df[~df.index.isin(test_df.index)]
    df.fillna(df.mean(), inplace=True)

    X = df.drop('Class', axis=1)
    y = df['Class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

    balanced_df = pd.concat([pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced, name='Class')], axis=1)
    #os.makedirs(os.path.dirname(TRANSFORMED_DATA_PATH), exist_ok=True)
    balanced_df.to_csv(TRANSFORMED_DATA_PATH, index=False)
    logging.info(f"Transformed data saved to {TRANSFORMED_DATA_PATH}")

if __name__ == '__main__':
    preprocess_data()