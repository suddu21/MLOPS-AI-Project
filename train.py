import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths (adjust these based on your environment)
TRANSFORMED_DATA_PATH = 'transformed_data.csv'
MODEL_PATH = 'models/fraud_model.pkl'
TEST_DATA_PATH = 'test_data.csv'

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
    """Train a RandomForestClassifier with hyperparameter grid search, using pre-saved test set, and save with scaler."""
    try:
        # Load preprocessed data
        print("Loading transformed dataset...")
        df = pd.read_csv(TRANSFORMED_DATA_PATH)
        print(f"Transformed dataset loaded successfully. Shape: {df.shape}")
        print(f"Class distribution:\n{df['Class'].value_counts()}")
        X = df.drop('Class', axis=1)
        y = df['Class']

        # Log the feature names and order
        feature_names = list(X.columns)
        print(f"Training feature names (order matters): {feature_names}")

        # Load pre-saved test set
        print("Loading pre-saved test set...")
        test_df = pd.read_csv(TEST_DATA_PATH)
        print(f"Test set loaded successfully. Shape: {test_df.shape}")
        print(f"Test set Class 1 count: {test_df['Class'].sum()}")
        X_test = test_df.drop('Class', axis=1)
        y_test = test_df['Class']

        # Train-validation split from transformed data
        print("\nSplitting transformed data into train and validation sets...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=42
        )
        print(f"Training set shape: {X_train.shape}")
        print(f"Validation set shape: {X_val.shape}")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [100],
            'max_depth': [10]
        }

        # Initialize the model
        base_model = RandomForestClassifier(random_state=42, verbose=1)

        # Perform grid search with cross-validation
        print("\nStarting hyperparameter grid search...")
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=2,
            scoring='f1',
            n_jobs=-1,
            verbose=2,
            refit=True,
        )

        # Fit grid search on scaled training data
        grid_search.fit(X_train_scaled, y_train)
        print("Grid search completed.")

        # Iterate over each hyperparameter combination
        for params, mean_score, std_score in zip(
            grid_search.cv_results_['params'],
            grid_search.cv_results_['mean_test_score'],
            grid_search.cv_results_['std_test_score']
        ):
            run_name = f"RF_n_estimators_{params['n_estimators']}_max_depth_{params['max_depth']}"
            with mlflow.start_run(run_name=run_name):
                print(f"\nEvaluating hyperparameter combination: {run_name}")
                print(f"Cross-validation F1-score: {mean_score:.4f} (+/- {std_score:.4f})")

                # Use the best estimator from grid search
                model = grid_search.best_estimator_

                # Evaluate on validation set
                print("Evaluating on validation set...")
                y_val_pred = model.predict(X_val_scaled)
                y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
                val_accuracy = accuracy_score(y_val, y_val_pred)
                val_precision = precision_score(y_val, y_val_pred)
                val_recall = recall_score(y_val, y_val_pred)
                val_f1 = f1_score(y_val, y_val_pred)
                val_auc = roc_auc_score(y_val, y_val_proba)
                val_balanced_accuracy = balanced_accuracy_score(y_val, y_val_pred)

                # Confusion matrix for validation set
                val_cm = plot_confusion_matrix(y_val, y_val_pred, f"Validation Confusion Matrix ({run_name})", f"val_confusion_matrix_{run_name}.png")
                print("Validation Confusion Matrix:")
                print(val_cm)

                # Evaluate on pre-saved test set
                print("Evaluating on pre-saved test set...")
                y_test_pred = model.predict(X_test_scaled)
                y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
                test_accuracy = accuracy_score(y_test, y_test_pred)
                test_precision = precision_score(y_test, y_test_pred)
                test_recall = recall_score(y_test, y_test_pred)
                test_f1 = f1_score(y_test, y_test_pred)
                test_auc = roc_auc_score(y_test, y_test_proba)
                test_balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)

                # Confusion matrix for test set
                test_cm = plot_confusion_matrix(y_test, y_test_pred, f"Test Confusion Matrix ({run_name})", f"test_confusion_matrix_{run_name}.png")
                print("Test Confusion Matrix:")
                print(test_cm)

                # Print evaluation metrics
                print("\nValidation Metrics:")
                print(f"Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}, Balanced Accuracy: {val_balanced_accuracy:.4f}")
                print("\nTest Metrics:")
                print(f"Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}, Balanced Accuracy: {test_balanced_accuracy:.4f}")

                # Log parameters and metrics to MLflow
                mlflow.log_param("n_estimators", params['n_estimators'])
                mlflow.log_param("max_depth", params['max_depth'])
                mlflow.log_metric("cv_f1_score", mean_score)
                mlflow.log_metric("cv_f1_score_std", std_score)
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
                mlflow.log_artifact(f"val_confusion_matrix_{run_name}.png")
                mlflow.log_artifact(f"test_confusion_matrix_{run_name}.png")
                mlflow.log_text(str(val_cm.tolist()), f"val_confusion_matrix_{run_name}.txt")
                mlflow.log_text(str(test_cm.tolist()), f"test_confusion_matrix_{run_name}.txt")

                # Log model
                mlflow.sklearn.log_model(model, "fraud_detection_model")

        # Train and save the best model with scaler
        print("\nTraining best model with optimal hyperparameters...")
        best_params = grid_search.best_params_
        best_model = RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            random_state=42
        )
        best_model.fit(X_train_scaled, y_train)

        # Evaluate best model on validation and test sets
        print("Evaluating best model on validation set...")
        y_val_pred = best_model.predict(X_val_scaled)
        y_val_proba = best_model.predict_proba(X_val_scaled)[:, 1]
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred)
        val_recall = recall_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        val_auc = roc_auc_score(y_val, y_val_proba)
        val_balanced_accuracy = balanced_accuracy_score(y_val, y_val_pred)

        print("Evaluating best model on test set...")
        y_test_pred = best_model.predict(X_test_scaled)
        y_test_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, y_test_proba)
        test_balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)

        # Log best model separately
        with mlflow.start_run(run_name="Best_Model"):
            mlflow.log_param("n_estimators", best_params['n_estimators'])
            mlflow.log_param("max_depth", best_params['max_depth'])
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
            mlflow.sklearn.log_model(best_model, "best_fraud_detection_model")

        # Save best model, scaler, and feature names as a dictionary using pickle
        print(f"\nSaving best model, scaler, and feature names to {MODEL_PATH}...")
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump({'model': best_model, 'scaler': scaler, 'feature_names': feature_names}, f)
        logging.info("Model training completed successfully")
        logging.info(f"Best Parameters: {best_params}")
        logging.info(f"Validation Metrics - Accuracy: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}, F1: {val_f1}, AUC: {val_auc}, Balanced Accuracy: {val_balanced_accuracy}")
        logging.info(f"Test Metrics - Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1: {test_f1}, AUC: {test_auc}, Balanced Accuracy: {test_balanced_accuracy}")
        print("Model training and evaluation completed successfully.")

    except Exception as e:
        logging.error(f"Model training failed: {str(e)}")
        print(f"Error: Model training failed - {str(e)}")
        raise

if __name__ == "__main__":
    train_model()