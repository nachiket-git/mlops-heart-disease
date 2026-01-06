import argparse
import os
import json
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from joblib import dump

from .data import load_and_validate, clean_basic
from .features import FeatureSpec, build_preprocess, split_xy

def evaluate_holdout(pipe, X_train, X_test, y_train, y_test):
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }

def cv_metrics(pipe, X, y, seed=42):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scoring = {"acc": "accuracy", "prec": "precision", "rec": "recall", "roc": "roc_auc"}
    res = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=None)
    return {
        "cv_accuracy_mean": float(res["test_acc"].mean()),
        "cv_precision_mean": float(res["test_prec"].mean()),
        "cv_recall_mean": float(res["test_rec"].mean()),
        "cv_roc_auc_mean": float(res["test_roc"].mean()),
    }

def run(csv_path: str, out_dir: str, experiment_name: str = "heart-disease-mlops"):
    df = load_and_validate(csv_path)
    df = clean_basic(df)

    spec = FeatureSpec()
    X, y = split_xy(df, spec)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pre = build_preprocess(spec)

    candidates = [
        ("logreg", LogisticRegression(max_iter=2000, solver="liblinear")),
        ("rf", RandomForestClassifier(
            n_estimators=400, max_depth=None, random_state=42, n_jobs=None
        )),
    ]

    os.makedirs(out_dir, exist_ok=True)

    mlflow.set_experiment(experiment_name)

    best = None
    best_auc = -1.0

    for name, model in candidates:
        pipe = Pipeline([
            ("preprocess", pre),
            ("model", model),
        ])

        with mlflow.start_run(run_name=name):
            # Params
            mlflow.log_param("model_name", name)
            mlflow.log_param("model_class", model.__class__.__name__)
            for k, v in model.get_params().items():
                # keep params small & serializable
                if isinstance(v, (str, int, float, bool)) or v is None:
                    mlflow.log_param(f"model__{k}", v)

            # CV metrics
            cvm = cv_metrics(pipe, X_train, y_train)
            for k, v in cvm.items():
                mlflow.log_metric(k, v)

            # Holdout metrics
            hm = evaluate_holdout(pipe, X_train, X_test, y_train, y_test)
            for k, v in hm.items():
                mlflow.log_metric(f"holdout_{k}", v)

            # Save model artifact in MLflow + joblib
            mlflow.sklearn.log_model(pipe, artifact_path="model")

            # also persist locally
            local_model_path = os.path.join(out_dir, f"{name}_pipeline.joblib")
            dump(pipe, local_model_path)
            mlflow.log_artifact(local_model_path, artifact_path="joblib")

            # track best by holdout roc_auc
            if hm["roc_auc"] > best_auc:
                best_auc = hm["roc_auc"]
                best = (name, pipe, hm, cvm)

    assert best is not None
    best_name, best_pipe, best_hm, best_cvm = best

    # Train final best on full data
    best_pipe.fit(X, y)
    final_path = os.path.join(out_dir, "final_pipeline.joblib")
    dump(best_pipe, final_path)

    # Write metrics summary
    summary = {
        "best_model": best_name,
        "best_holdout": best_hm,
        "best_cv": best_cvm,
        "final_model_path": final_path,
    }
    with open(os.path.join(out_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--csv", required=True)
    r.add_argument("--out", default="artifacts/model")
    r.add_argument("--experiment", default="heart-disease-mlops")
    args = p.parse_args()

    if args.cmd == "run":
        run(args.csv, args.out, args.experiment)

if __name__ == "__main__":
    main()
