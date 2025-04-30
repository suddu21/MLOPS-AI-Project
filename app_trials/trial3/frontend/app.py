import streamlit as st
import pandas as pd
import requests
import json
import os

# Set page configuration for a wider layout and custom title
st.set_page_config(page_title="Fraudulent Transaction Detector", layout="wide")

# Initialize session state variables
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = None

# Sidebar for app info
st.sidebar.title("About")
st.sidebar.write("**Fraudulent Transaction Detector**")
st.sidebar.write("**Maintainer:** Sudhanva Satish DA24M023")
st.sidebar.write("---")

# Main content
st.title("Fraudulent Transaction Detector")
st.write("### Welcome!")
st.write("""
**Instructions:**
- Upload a CSV file containing transaction data with columns: `Time`, `Amount`, `V1` to `V28`.
- Ensure the data is formatted with these columns in the same order as used during training.
- Click the 'Predict' button to detect fraudulent transactions.
- Review the predictions and use checkboxes to mark incorrect predictions (check the box if the prediction is wrong).
- Click 'Save Feedback' to record your feedback for future model retraining.
""")
st.write("---")

# File uploader section
uploaded_file = st.file_uploader("Upload your CSV file", type="csv", key="file_uploader")
if uploaded_file is not None:
    # Reset file pointer to the beginning
    uploaded_file.seek(0)
    # Read and display the file content for debugging
    content = uploaded_file.read().decode('utf-8')
    #st.write("**Raw File Content:**", content)
    # Convert to DataFrame to verify parsing
    try:
        df = pd.read_csv(pd.io.common.StringIO(content))
        st.session_state.uploaded_df = df
        st.write("**Uploaded Data (Preview):**", df.head())
    except Exception as e:
        st.error(f"Failed to parse CSV: {str(e)}")
        st.stop()

# Predict button
if st.button("Predict", key="predict_button"):
    if uploaded_file is not None:
        # Reset file pointer again before sending to backend
        uploaded_file.seek(0)
        files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
        try:
            with st.spinner("Processing prediction..."):
                response = requests.post("http://backend:8000/predict", files=files, timeout=30)
            if response.status_code == 200:
                result = response.json()
                predictions_df = pd.DataFrame(result["predictions"])
                # Add feedback column with default unchecked
                predictions_df['Feedback Correct'] = False  # Default to False (unchecked)
                # Store predictions in session state
                st.session_state.predictions = predictions_df
                st.session_state.feedback_data = predictions_df.copy()
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Request failed: {str(e)}")
    else:
        st.error("Please upload a CSV file before predicting.")

# Display predictions and collect feedback if predictions exist
if st.session_state.predictions is not None:
    st.success("Prediction completed!")
    st.write("**Prediction Results:**")
    # Display with checkboxes for feedback
    edited_df = st.data_editor(
        st.session_state.predictions,
        column_config={
            "Feedback Correct": st.column_config.CheckboxColumn(
                "Mark as Incorrect",
                help="Check if the prediction is incorrect",
                default=False
            )
        },
        disabled=["row", "data", "prediction", "fraud_probability"],
        hide_index=True,
        key="feedback_editor"
    )
    # Update feedback data in session state
    st.session_state.feedback_data = edited_df

    # Save feedback button
    if st.button("Save Feedback", key="save_feedback"):
        # Prepare feedback data
        feedback_data = st.session_state.feedback_data.copy()
        feedback_data['original_data'] = feedback_data['data'].apply(json.dumps)
        feedback_data = feedback_data.drop(columns=['data'])
        feedback_data['feedback_correct'] = (feedback_data['Feedback Correct'] == False).astype(int)
        feedback_data = feedback_data.drop(columns=['Feedback Correct'])

        # Define the container path for the feedback CSV
        feedback_dir = "/app/data"
        feedback_csv_path = os.path.join(feedback_dir, "raw_feedback.csv")

        # Ensure the directory exists
        if not os.path.exists(feedback_dir):
            os.makedirs(feedback_dir)

        # Check if file exists and append or create
        file_exists = os.path.isfile(feedback_csv_path)
        feedback_data.to_csv(feedback_csv_path, mode='a', header=not file_exists, index=False)
        st.success(f"Feedback saved to {feedback_csv_path}. Total rows: {len(feedback_data)}")
        # Display summary
        st.write("**Summary:**")
        fraud_count = edited_df[edited_df['prediction'] == 'Fraud'].shape[0]
        total_rows = edited_df.shape[0]
        incorrect_count = edited_df[edited_df['Feedback Correct']].shape[0]
        st.write(f"Total Rows: {total_rows}")
        st.write(f"Fraudulent Transactions: {fraud_count} ({(fraud_count/total_rows*100):.2f}%)")
        st.write(f"Incorrect Predictions: {incorrect_count} ({(incorrect_count/total_rows*100):.2f}%)")
        # Reset session state after saving feedback
        st.session_state.predictions = None
        st.session_state.feedback_data = None
        st.session_state.uploaded_df = None
        st.experimental_rerun()  # Force a rerun to reset the app state
else:
    if st.session_state.uploaded_df is not None:
        st.write("Click 'Predict' to see the results.")