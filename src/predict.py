from typing import Dict, Any, Tuple
import numpy as np
from joblib import load

MODEL_PATH_DEFAULT = "artifacts/model/final_pipeline.joblib"

def load_model(path: str = MODEL_PATH_DEFAULT):
    return load(path)

def predict_one(model, features: Dict[str, Any]) -> Tuple[int, float]:
    # sklearn pipeline expects 2D tabular input; dict -> DataFrame
    import pandas as pd
    X = pd.DataFrame([features])
    proba = float(model.predict_proba(X)[:, 1][0])
    pred = int(proba >= 0.5)
    return pred, proba
