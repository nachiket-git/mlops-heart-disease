import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from src.features import FeatureSpec, build_preprocess, split_xy
from src.predict import predict_one

def test_predict_one_runs():
    df = pd.DataFrame({
        "age":[63,37,41,56],
        "sex":[1,1,0,1],
        "cp":[3,2,1,1],
        "trestbps":[145,130,130,120],
        "chol":[233,250,204,236],
        "fbs":[1,0,0,0],
        "restecg":[0,1,0,1],
        "thalach":[150,187,172,178],
        "exang":[0,0,0,0],
        "oldpeak":[2.3,3.5,1.4,0.8],
        "slope":[0,0,2,2],
        "ca":[0,0,0,0],
        "thal":[1,2,2,2],
        "target":[1,1,0,0],
    })
    spec = FeatureSpec()
    X, y = split_xy(df, spec)
    pipe = Pipeline([("preprocess", build_preprocess(spec)), ("model", LogisticRegression(max_iter=1000, solver="liblinear"))])
    pipe.fit(X, y)

    pred, proba = predict_one(pipe, X.iloc[0].to_dict())
    assert pred in (0,1)
    assert 0.0 <= proba <= 1.0
