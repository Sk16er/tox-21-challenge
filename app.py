from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn
from train import TASKS
from predict import Tox21Predictor
import os

app = FastAPI(
    title="Tox21 Toxicity Prediction API (Ensemble)",
    description="Predicts toxicity across 12 endpoints using an ensemble of D-MPNNs with RDKit descriptors.",
    version="2.0.0"
)

# Global predictor instance
predictor = None

class PredictionRequest(BaseModel):
    smiles: List[str]

@app.on_event("startup")
def load_model():
    global predictor
    # Tox21Predictor now handles finding all model_seed_*.pt files
    predictor = Tox21Predictor()

@app.post("/predict")
def predict(request: PredictionRequest) -> Dict[str, Any]:
    """
    Accepts { "smiles": [ ... ] }
    Returns { "SMILES": { "Task": prob ... }, ... }
    """
    if predictor is None or not predictor.models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    results = predictor.predict(request.smiles)
    return results

@app.get("/metadata")
def get_metadata():
    model_count = len(predictor.models) if predictor else 0
    return {
        "model": "D-MPNN Ensemble (RDKit Augmented)",
        "tasks": TASKS,
        "ensemble_size": model_count,
        "version": "2.0.0",
        "author": "sk16er",
        "split": "Scaffold (90/10 Train/Val)"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
