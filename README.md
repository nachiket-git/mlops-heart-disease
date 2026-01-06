# Heart Disease Risk Classifier (MLOps Assignment I)

End-to-end, production-style ML pipeline:
- Data download + cleaning + EDA
- Feature engineering (ColumnTransformer) + model training (LogReg, RandomForest)
- Experiment tracking with **MLflow**
- Packaging a reproducible preprocessing+model pipeline
- FastAPI inference service (`/predict`, `/health`, `/metrics`)
- Docker containerization
- Kubernetes manifests (Minikube-ready)
- CI with GitHub Actions (lint + unit tests + train smoke-run)

## Quickstart (local)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt

# 1) Download dataset
python -m src.data download --out data/raw/heart.csv

# 2) Run EDA (generates plots)
python -m src.eda run --csv data/raw/heart.csv --out artifacts/eda

# 3) Train models with MLflow tracking (logs metrics + artifacts)
mlflow ui --host 127.0.0.1 --port 5000
python -m src.train run --csv data/raw/heart.csv --out artifacts/model

# 4) Run API
uvicorn src.api.main:app --reload --port 8000

# Test
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

## Docker

```bash
docker build -t heart-mlops:latest .
docker run -p 8000:8000 heart-mlops:latest
```

## Kubernetes (Minikube)

```bash
minikube start
kubectl apply -f k8s/
minikube service heart-mlops-svc
```

## Repo layout

- `src/` : python modules (download, EDA, training, API)
- `tests/` : unit tests (pytest)
- `.github/workflows/ci.yml` : CI pipeline
- `k8s/` : Kubernetes manifests (deployment + service)
- `artifacts/` : generated models/plots (ignored by git in real repo)
