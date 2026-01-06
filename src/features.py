from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

@dataclass(frozen=True)
class FeatureSpec:
    target_col: str = "target"
    numeric_cols: Tuple[str, ...] = (
        "age","trestbps","chol","thalach","oldpeak","ca"
    )
    categorical_cols: Tuple[str, ...] = (
        "sex","cp","fbs","restecg","exang","slope","thal"
    )

def build_preprocess(spec: FeatureSpec) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, list(spec.numeric_cols)),
        ("cat", cat_pipe, list(spec.categorical_cols)),
    ])
    return pre

def split_xy(df: pd.DataFrame, spec: FeatureSpec):
    X = df.drop(columns=[spec.target_col])
    y = df[spec.target_col].astype(int).values
    return X, y
