🔹 Project Title

Credit Card Fraud Detection System (ML + FastAPI + Docker)

🔹 Overview

This project is an end-to-end fraud detection system built using classical machine learning models and deployed as a production-style inference service.

The system:

Trains and evaluates multiple fraud detection models

Converts probabilistic outputs into business decisions (BLOCK / ALLOW)

Serves predictions via a FastAPI REST API

Is fully containerized using Docker for reproducible deployment

The design is inspired by real-world fintech systems (e.g., Stripe-style risk engines) where model outputs are separated from decision logic.

🔹 Problem Statement

Credit card fraud detection is a highly imbalanced classification problem, where fraudulent transactions are rare but extremely costly.

Key challenges:

Severe class imbalance

Need for high recall at low false-positive rates

Requirement for explainable, controllable decisioning

Production constraints (latency, reproducibility, reliability)

This project addresses these challenges using a multi-model, threshold-based risk scoring approach.

🔹 Dataset

Dataset: European Credit Card Transactions

Source: Public Kaggle dataset

Records: ~285,000 transactions

Fraud rate: ~0.17%

Features:

V1–V28: PCA-transformed transaction features

Time: Seconds since first transaction

Amount: Transaction amount

Class: Fraud label (1 = fraud, 0 = legitimate)


🔹 System Architecture

Incoming Transaction (JSON)
        |
        v
Feature Validation
        |
        v
Preprocessing
(StandardScaler for LR)
        |
        v
Model Scoring
(LR / RF / Ensemble)
        |
        v
Threshold-based Decision
(BLOCK / ALLOW)
        |
        v
FastAPI Response

🔹 Models Used
Model	Purpose
Logistic Regression	Interpretable, stable baseline
Random Forest	Non-linear pattern capture
Ensemble	Weighted combination of LR + RF

The system supports runtime model switching:

lr → Logistic Regression

rf → Random Forest

ensemble → Combined score

🔹 Decision Logic

Models output a fraud probability, not a decision.

Final decisions are made using a configurable threshold:

fraud_score ≥ threshold  → BLOCK
fraud_score < threshold  → ALLOW

This allows business teams to tune risk tolerance without retraining models.

Example:

Lower threshold → higher fraud catch rate, more false positives

Higher threshold → fewer false positives, more fraud leakage

🔹 Evaluation Metrics

Fraud detection is evaluated using appropriate metrics for imbalanced data:

ROC-AUC

Precision–Recall AUC

Precision / Recall at selected thresholds

Emphasis is placed on recall under constrained false-positive rates, aligning with real-world fraud systems.

(Exact metrics can be added here if you want to include numbers)

🔹 API Endpoints

Health Check
GET /health

Fraud Prediction
POST /predict

Example request:
{
  "transaction": {
    "Time": 0.0,
    "V1": -1.359807,
    "V2": -0.072781,
    "...": "...",
    "Amount": 149.62
  },
  "model": "ensemble",
  "threshold": 0.1
}

Example response:
{
  "fraud_score": 0.032,
  "decision": "ALLOW",
  "used_model": "ensemble",
  "threshold": 0.1
}


🔹 Project Structure

fraud_engine/
├── models/
│   ├── scaler.joblib
│   ├── lr_model.joblib
│   ├── rf_model.joblib
│   └── feature_columns.json
│
├── src/
│   └── api/
│       ├── main.py
│       └── schemas.py
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── Dockerfile
├── requirements.txt
└── README.md

🔹 Running the Project (Docker)

Build the image
docker build -t fraud-engine .

Run the container:
docker run -p 8000:8000 fraud-engine

Access API

Health: http://127.0.0.1:8000/health

Swagger UI: http://127.0.0.1:8000/docs

🔹 Key Design Decisions

Threshold-based decisioning instead of fixed probability cutoff

Model switching and ensemble support for experimentation

Separation of ML logic and business policy

Containerized deployment for reproducibility

🔹 Future Improvements

Add experiment tracking (MLflow)

Add request logging and monitoring

Implement data drift detection

Add CI/CD pipeline

Deploy on cloud infrastructure

🔹 Why This Project Matters

This project demonstrates:

Practical machine learning modeling

Production-style ML system design

API-based inference

MLOps fundamentals (Docker, reproducibility)

Business-aware decision making