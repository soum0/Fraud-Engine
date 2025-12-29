from pydantic import BaseModel 
from typing import Dict, Optional

class PredictRequest(BaseModel):
    transaction : Dict[str, float]
    model : Optional[str] = 'rf'
    threshold : Optional[float] = None

class PredictResponse(BaseModel):
    fraud_score : float 
    decision : str
    used_model : str
    threshold : float

