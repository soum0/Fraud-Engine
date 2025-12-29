# Credit Card Fraud Detection Engine

An end-to-end fraud detection system that combines machine learning modeling with production-grade API deployment using FastAPI and Docker.


This project focuses on **real-world fraud decisioning**, not just model training.

---

## 🚀 Key Features

- Fraud probability prediction using ML models
- Threshold-based business decisioning (BLOCK / ALLOW)
- Runtime model switching (LR / RF / Ensemble)
- REST API built with FastAPI
- Fully containerized using Docker

---

## 🧠 Problem Overview

Credit card fraud detection is a **highly imbalanced classification problem**, where fraudulent transactions are rare but costly.

Challenges addressed:
- Extreme class imbalance
- Need for high recall with low false positives
- Clear separation of ML scoring and business decisions
- Production-ready deployment

---

## 📊 Dataset

- European credit card transaction dataset
- ~285,000 transactions
- Fraud rate: ~0.17%
- Features:
  - `V1–V28`: PCA-transformed features
  - `Time`: Time since first transaction
  - `Amount`: Transaction amount

---

## 🏗 System Architecture

Transaction JSON
↓
Feature Validation
↓
Preprocessing
(LR → Scaled | RF → Raw)
↓
Model Scoring
↓
Threshold Decision
(BLOCK / ALLOW)
↓
API Response



---

## 🤖 Models Used

| Model | Purpose |
|-----|--------|
| Logistic Regression | Interpretable baseline |
| Random Forest | Non-linear pattern detection |
| Ensemble | Weighted LR + RF scoring |

Supported modes:
- `lr`
- `rf`
- `ensemble`

---

## ⚖ Decision Logic

Models output a **fraud probability**, not a final decision.



Final decisions are made using a configurable threshold:

fraud_score ≥ threshold  → BLOCK
fraud_score < threshold  → ALLOW


This allows risk tolerance to be adjusted **without retraining models**.

---

## 🌐 API Endpoints

### Health Check
GET /health


### Predict Fraud
POST /predict



Example request:
```json
{
  "transaction": { "...": "features" },
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



🗂 Project Structure

fraud_engine/
├── README.md
├── 01_eda.ipynb
├── src/
│   └── api/
│       ├── main.py
│       └── schemas.py
├── models/
├── Dockerfile
└── requirements.txt


🐳 Running with Docker

Build image:
docker build -t fraud-engine .

Run container:
docker run -p 8000:8000 fraud-engine


Access:

API: http://127.0.0.1:8000

Docs: http://127.0.0.1:8000/docs


🧩 Design Highlights

Separation of ML scoring and decision logic

Model-agnostic inference API

Production-style project structure

Reproducible deployment using Docker

🔮 Future Improvements

Model monitoring and drift detection

Experiment tracking (MLflow)

CI/CD pipeline

Cloud deployment

🎯 Why This Project Matters

This project demonstrates:

Applied machine learning

ML system design

API-based inference

MLOps fundamentals

Business-aware decisioning