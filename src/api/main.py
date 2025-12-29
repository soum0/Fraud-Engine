from fastapi import FastAPI, HTTPException
from src.api.schemas import PredictRequest, PredictResponse
import json
import os
import joblib
import pandas as pd





app = FastAPI( tittle = " Fraud Engine API")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FEATURE_PATH = os.path.join(BASE_DIR, "models", "feature_columns.json")

with open(FEATURE_PATH, "r") as f:
    FEATURE_COLUMNS = json.load(f)


SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")

scaler = joblib.load(SCALER_PATH)

LR_MODEL_PATH = os.path.join(BASE_DIR, "models", "lr_model.joblib")
lr_model = joblib.load(LR_MODEL_PATH)


RF_MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.joblib")
rf_model = joblib.load(RF_MODEL_PATH)



@app.get("/")
def root():
    return {"message": "Fraud API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/scaler-info")
def scaler_info():
    return {
        "scaler_loaded": True,
        "scaler_type": type(scaler).__name__
    }


def validate_transaction_features(transaction: dict):
    missing = [f for f in FEATURE_COLUMNS if f not in transaction]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing required features",
                "missing_features": missing
            }
        )
    
def prepare_scaled_input(transaction: dict):
    """
    Convert transaction dict to scaled numpy array
    using the same feature order as training.
    """
    # ensure correct feature order
    row = [transaction[feature] for feature in FEATURE_COLUMNS]

    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    X_scaled = scaler.transform(df)

    return X_scaled


def prepare_rf_input(transaction: dict):
    """
    Prepare raw (unscaled) input for Random Forest
    """
    row = [transaction[feature] for feature in FEATURE_COLUMNS]
    return [row]  # 2D array shape (1, n_features)



# @app.post("/predict", response_model=PredictResponse)
# def predict(request: PredictRequest):

#     validate_transaction_features(request.transaction)

#     X_scaled = prepare_scaled_input(request.transaction)

#     # Temporary debug print (safe)
#     print("Scaled input shape:", X_scaled.shape)

#     return PredictResponse(
#         fraud_score=0.0,
#         decision="ALLOW",
#         used_model=request.model,
#         threshold=request.threshold if request.threshold is not None else 0.5
#     )

# @app.post("/predict", response_model=PredictResponse)
# def predict(request: PredictRequest):

#     validate_transaction_features(request.transaction)

#     X_scaled = prepare_scaled_input(request.transaction)

#     # 🔥 REAL prediction
#     fraud_score = float(lr_model.predict_proba(X_scaled)[0][1])

#     return PredictResponse(
#         fraud_score=fraud_score,
#         decision="ALLOW",  # still dummy decision
#         used_model="logistic_regression",
#         threshold=request.threshold if request.threshold is not None else 0.5
#     )

# @app.post("/predict", response_model=PredictResponse)
# def predict(request: PredictRequest):

#     validate_transaction_features(request.transaction)

#     X_scaled = prepare_scaled_input(request.transaction)

#     fraud_score = float(lr_model.predict_proba(X_scaled)[0][1])

#     threshold = request.threshold if request.threshold is not None else 0.1

#     decision = "BLOCK" if fraud_score >= threshold else "ALLOW"

#     return PredictResponse(
#         fraud_score=fraud_score,
#         decision=decision,
#         used_model="logistic_regression",
#         threshold=threshold
#     )


# @app.post("/predict", response_model=PredictResponse)
# def predict(request: PredictRequest):

#     validate_transaction_features(request.transaction)

#     model_choice = request.model.lower() if request.model else "lr"
#     threshold = request.threshold if request.threshold is not None else 0.1

#     if model_choice == "lr":
#         X_scaled = prepare_scaled_input(request.transaction)
#         fraud_score = float(lr_model.predict_proba(X_scaled)[0][1])
#         used_model = "logistic_regression"

#     elif model_choice == "rf":
#         X_rf = prepare_rf_input(request.transaction)
#         fraud_score = float(rf_model.predict_proba(X_rf)[0][1])
#         used_model = "random_forest"

#     else:
#         raise HTTPException(status_code=400, detail="model must be 'lr' or 'rf'")

#     decision = "BLOCK" if fraud_score >= threshold else "ALLOW"

#     return PredictResponse(
#         fraud_score=fraud_score,
#         decision=decision,
#         used_model=used_model,
#         threshold=threshold
#     )

############## ADDING ENSEMBLE MODEL IN PREDICTION ENDPOINT ##############

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):

    validate_transaction_features(request.transaction)

    model_choice = request.model.lower() if request.model else "lr"
    threshold = request.threshold if request.threshold is not None else 0.1

    if model_choice == "lr":
        X_scaled = prepare_scaled_input(request.transaction)
        fraud_score = float(lr_model.predict_proba(X_scaled)[0][1])
        used_model = "logistic_regression"

    elif model_choice == "rf":
        X_rf = prepare_rf_input(request.transaction)
        fraud_score = float(rf_model.predict_proba(X_rf)[0][1])
        used_model = "random_forest"

    elif model_choice == "ensemble":
        X_scaled = prepare_scaled_input(request.transaction)
        X_rf = prepare_rf_input(request.transaction)

        lr_score = float(lr_model.predict_proba(X_scaled)[0][1])
        rf_score = float(rf_model.predict_proba(X_rf)[0][1])

        fraud_score = 0.4 * lr_score + 0.6 * rf_score
        used_model = "ensemble"

    else:
        raise HTTPException(status_code=400, detail="model must be 'lr', 'rf', or 'ensemble'")

    decision = "BLOCK" if fraud_score >= threshold else "ALLOW"

    return PredictResponse(
        fraud_score=fraud_score,
        decision=decision,
        used_model=used_model,
        threshold=threshold
    )



