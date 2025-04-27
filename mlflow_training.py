import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score
import pandas as pd
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define paths (adjust these based on your environment)
TRANSFORMED_DATA_PATH = 'transformed_data.csv'
MODEL_PATH = 'models/fraud_model.joblib'

def plot_confusion_matrix(y_true, y_pred, title, filename):
    """Generate and save a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(filename)
    plt.close()
    return cm

def train_model():
    """Train a RandomForestClassifier with stratified sampling, validation testing, and enhanced metrics."""
    try:
        # Load preprocessed data
        print("Loading transformed dataset...")
        df = pd.read_csv(TRANSFORMED_DATA_PATH)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        print(f"Class distribution:\n{df['Class'].value_counts()}")
        X = df.drop('Class', axis=1)
        y = df['Class']

        # Stratified train-validation-test split
        print("\nSplitting dataset into train, validation, and test sets...")
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42
        )
        print(f"Training set shape: {X_train.shape}")
        print(f"Validation set shape: {X_val.shape}")
        print(f"Test set shape: {X_test.shape}")

        # Start MLflow run
        with mlflow.start_run():
            # Train model
            print("\nTraining RandomForestClassifier...")
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            model.fit(X_train, y_train)
            print("Training completed.")

            # Evaluate on validation set
            print("\nEvaluating on validation set...")
            y_val_pred = model.predict(X_val)
            y_val_proba = model.predict_proba(X_val)[:, 1]
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_precision = precision_score(y_val, y_val_pred)
            val_recall = recall_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred)
            val_auc = roc_auc_score(y_val, y_val_proba)
            val_balanced_accuracy = balanced_accuracy_score(y_val, y_val_pred)

            # Confusion matrix for validation set
            val_cm = plot_confusion_matrix(y_val, y_val_pred, "Validation Confusion Matrix", "val_confusion_matrix.png")
            print("Validation Confusion Matrix:")
            print(val_cm)

            # Evaluate on test set
            print("\nEvaluating on test set...")
            y_test_pred = model.predict(X_test)
            y_test_proba = model.predict_proba(X_test)[:, 1]
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred)
            test_recall = recall_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred)
            test_auc = roc_auc_score(y_test, y_test_proba)
            test_balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)

            # Confusion matrix for test set
            test_cm = plot_confusion_matrix(y_test, y_test_pred, "Test Confusion Matrix", "test_confusion_matrix.png")
            print("Test Confusion Matrix:")
            print(test_cm)

            # Print evaluation metrics
            print("\nValidation Metrics:")
            print(f"Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}, Balanced Accuracy: {val_balanced_accuracy:.4f}")
            print("\nTest Metrics:")
            print(f"Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}, Balanced Accuracy: {test_balanced_accuracy:.4f}")

            # Log parameters and metrics to MLflow
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)
            mlflow.log_metric("val_accuracy", val_accuracy)
            mlflow.log_metric("val_precision", val_precision)
            mlflow.log_metric("val_recall", val_recall)
            mlflow.log_metric("val_f1_score", val_f1)
            mlflow.log_metric("val_auc", val_auc)
            mlflow.log_metric("val_balanced_accuracy", val_balanced_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("test_precision", test_precision)
            mlflow.log_metric("test_recall", test_recall)
            mlflow.log_metric("test_f1_score", test_f1)
            mlflow.log_metric("test_auc", test_auc)
            mlflow.log_metric("test_balanced_accuracy", test_balanced_accuracy)

            # Log confusion matrices as artifacts
            mlflow.log_artifact("val_confusion_matrix.png")
            mlflow.log_artifact("test_confusion_matrix.png")
            mlflow.log_text(str(val_cm.tolist()), "val_confusion_matrix.txt")
            mlflow.log_text(str(test_cm.tolist()), "test_confusion_matrix.txt")

            # Log model
            mlflow.log_artifact(MODEL_PATH, "model")
            mlflow.sklearn.log_model(model, "fraud_detection_model")

            # Save model for inference
            print(f"\nSaving model to {MODEL_PATH}...")
            joblib.dump(model, MODEL_PATH)
            logging.info("Model training completed successfully")
            logging.info(f"Validation Metrics - Accuracy: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}, F1: {val_f1}, AUC: {val_auc}, Balanced Accuracy: {val_balanced_accuracy}")
            logging.info(f"Test Metrics - Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1: {test_f1}, AUC: {test_auc}, Balanced Accuracy: {test_balanced_accuracy}")
            print("Model training and evaluation completed successfully.")

    except Exception as e:
        logging.error(f"Model training failed: {str(e)}")
        print(f"Error: Model training failed - {str(e)}")
        raise

if __name__ == "__main__":
    train_model()