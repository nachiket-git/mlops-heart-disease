import pandas as pd
import pytest
from src.data import clean_basic

def test_clean_basic_imputes_and_binary_target():
    df = pd.DataFrame({
        "age":[63,None],
        "sex":[1,0],
        "cp":[3,2],
        "trestbps":[145,120],
        "chol":[233,250],
        "fbs":[1,0],
        "restecg":[0,1],
        "thalach":[150,160],
        "exang":[0,1],
        "oldpeak":[2.3,1.4],
        "slope":[0,1],
        "ca":[0,2],
        "thal":[1,2],
        "target":[4,0],  # non-binary in first row
    })
    out = clean_basic(df)
    assert out.isna().sum().sum() == 0
    assert set(out["target"].unique()) <= {0,1}
