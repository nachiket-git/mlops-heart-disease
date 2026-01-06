# MLOps Assignment I Report (Heart Disease Risk Prediction)

> Replace bracketed placeholders with your own screenshots and repo links.

## 1. Objective
Build a machine learning classifier to predict heart disease risk and deploy it as a cloud-ready, monitored API.

## 2. Dataset
Heart Disease (UCI). We use a processed CSV (14 columns incl. target) downloaded by `src/data.py`.

## 3. Solution Architecture
**Offline pipeline**
1. Download + validate + clean
2. EDA + plots
3. Training (LogReg + RandomForest)
4. Experiment tracking (MLflow)
5. Model packaging (joblib pipeline)

**Online serving**
- FastAPI service with `/predict`, `/health`, `/metrics`
- Docker container
- Kubernetes deployment (minikube or cloud K8s)

[Insert architecture diagram screenshot]

## 4. EDA
- Class balance plot
- Histograms
- Correlation heatmap

[Insert screenshots from `artifacts/eda/`]

## 5. Feature Engineering
- Numeric: StandardScaler
- Categorical: OneHotEncoder
- Combined using ColumnTransformer to guarantee reproducibility.

## 6. Model Development
Models:
- Logistic Regression baseline
- Random Forest (stronger non-linear baseline)

Evaluation:
- 5-fold stratified CV: accuracy, precision, recall, ROC-AUC
- Holdout test metrics

[Insert table of metrics]

## 7. Experiment Tracking (MLflow)
- Parameters logged for each run
- Metrics logged (CV + holdout)
- Artifacts logged (trained pipeline)

[Insert MLflow UI screenshots]

## 8. Packaging & Reproducibility
- Single pipeline includes preprocessing + estimator
- Saved as `final_pipeline.joblib`
- Environment pinned in `requirements.txt`

## 9. CI/CD
GitHub Actions workflow:
- ruff lint
- pytest unit tests
- smoke train step

[Insert GH Actions run screenshots]

## 10. Containerization
- Dockerfile builds image and serves API on 8000
- Local run + cURL test

[Insert docker build/run screenshots]

## 11. Deployment (Kubernetes)
- Deployment + Service manifests in `k8s/`
- Expose service via NodePort or Ingress
- Verify `/health` and `/predict`

[Insert kubectl and endpoint screenshots]

## 12. Monitoring & Logging
- Request logging via python logging
- Prometheus metrics endpoint `/metrics` and example scrape config

[Insert monitoring screenshots]

## 13. Repository & How to Run
- Repo link: [your link]
- Setup steps: see README
