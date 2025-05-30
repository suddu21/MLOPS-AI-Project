import pandas as pd
import ast
import json
import re
import logging

logging.basicConfig(level=logging.INFO,
                    handlers=[logging.FileHandler(r"C:\Users\sudha\Desktop\My-Data\Education\IITM\2nd-sem\ML-Ops-Lab\MLOPS-AI-Project\dvc_src\logs\transform_feedback.log"), logging.StreamHandler()]
)

# Read the original CSV 
logging.info
df = pd.read_csv(r'C:\Users\sudha\Desktop\My-Data\Education\IITM\2nd-sem\ML-Ops-Lab\MLOPS-AI-Project\dvc_src\data\raw_feedback.csv')

# Create a new dataframe to store the transformed data
transformed_data = []

# Process each row
for _, row in df.iterrows():
    # Extract the original data string and convert it to a dictionary
    # The original data is in a format that's not quite JSON, so we need to clean it
    original_data_str = row['original_data']
    
    # Remove the triple quotes and convert Python-style dict to JSON format
    cleaned_str = original_data_str.strip('\"\"\"').replace("'", "\"")
    
    # Parse the JSON string
    try:
        original_data = json.loads(cleaned_str)
    except json.JSONDecodeError:
        # If JSON parsing fails, try using ast.literal_eval
        try:
            original_data = ast.literal_eval(cleaned_str)
        except:
            print(f"Failed to parse row: {row['row']}")
            continue
    
    # Get the prediction and feedback_correct values
    prediction = row['prediction']
    feedback_correct = row['feedback_correct']
    
    # Determine the Class value based on the mapping rules
    if feedback_correct == 1:
        class_value = 0 if prediction == 'Not Fraud' else 1
    else:  # feedback_correct == 0
        class_value = 1 if prediction == 'Not Fraud' else 0
    
    # Add all fields from original_data plus the Class
    row_data = original_data.copy()
    row_data['Class'] = class_value
    
    transformed_data.append(row_data)

logging.info(f"Transformed {len(transformed_data)} rows of feedback data.")

# Convert to DataFrame
result_df = pd.DataFrame(transformed_data)

# Save to CSV
result_df.to_csv(r'C:\Users\sudha\Desktop\My-Data\Education\IITM\2nd-sem\ML-Ops-Lab\MLOPS-AI-Project\dvc_src\data\transformed_feedback.csv', index=False)

logging.info("Transformed feedback data saved to transformed_feedback.csv")

#print("Transformation complete. Output saved to 'transformed_feedback.csv'")