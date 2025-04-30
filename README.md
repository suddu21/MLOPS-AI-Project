# Fraudulent Transaction Detection System

A Machine Learning Operations (MLOps) project that implements a fraud detection system for financial transactions using machine learning and ML Ops practices.

## Project Overview

This project implements a robust fraud detection system that:
- Uses machine learning to identify potentially fraudulent transactions
- Implements MLOps best practices for model training and deployment
- Provides a web interface for real-time transaction monitoring
- Includes comprehensive documentation and evaluation guidelines

## Key Components

- **ML Pipeline**: Automated training and evaluation pipeline using DVC
- **Web Application**: User-friendly interface for transaction monitoring
- **Model Management**: Version control for ML models and experiments
- **Documentation**: Detailed HLD, LLD, and user manual

## Project Structure

```
├── AIProject_Web_App/        # Web application code
├── dvc_src/                  # DVC pipeline source code
├── models/                   # Trained model artifacts
├── mlruns/                   # MLflow experiment tracking
├── mydvc/                    # DVC configuration
└── docs/                     # Project documentation
```

## Getting Started

1. Clone the repository
2. Install dependencies
3. Run the DVC pipeline
4. Start the web application

## Documentation

- [High-Level Design (HLD)](HLD_Fraudulent_Transaction_Detector.pdf)
- [Low-Level Design (LLD)](LLD_Fraudulent_Transaction_Detector.pdf)
- [User Manual](Fraudulent_Transaction_Detector_User_Manual.pdf)
- [Evaluation Guidelines](AI%20Application%20Evaluation%20Guideline.pdf)

## Technologies Used

- Python
- DVC (Data Version Control)
- MLflow
- Web Framework (Flask/Django)
- Machine Learning Libraries

## License

This project is licensed under the MIT License - see the LICENSE file for details.
