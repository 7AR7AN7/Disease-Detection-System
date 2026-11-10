"""Training pipeline for the multi-disease detection API.

Steps for each disease (diabetes, heart, liver):
  1. Download the public dataset (cached in ./data/).
  2. Clean / prepare the data (missing values, label binarisation, encoding).
  3. Build sklearn pipelines (imputation, scaling, one-hot, log transform) for several
     candidate classifiers and select the best by 5-fold cross-validated F1.
  4. Evaluate the selected model on a held-out stratified 20% test split
     (accuracy, precision, recall, F1, ROC-AUC, confusion matrix).
  5. Refit on the full dataset and save the pipeline to ./models/<disease>.joblib.

All metrics are written to ./models/metrics.json and printed to the console.

Usage:
    .venv/bin/python3 train_models.py
"""

import json
import logging
import os
import sys
import urllib.request
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from app.specs import DISEASES  # noqa: E402

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train")


# --------------------------------------------------------------------------- data
def download(disease: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{disease}.csv")
    if not os.path.exists(path):
        url = DISEASES[disease]["url"]
        logger.info("Downloading %s dataset from %s", disease, url)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(path, "wb") as f:
            f.write(resp.read())
    return path


def _read_rows(path: str):
    with open(path) as f:
        return [line.strip().split(",") for line in f if line.strip()]


def _num(v: str) -> float:
    v = v.strip()
    return np.nan if v in ("", "?", "NA") else float(v)


def load_diabetes(path):
    rows = _read_rows(path)
    data = np.array([[_num(v) for v in r] for r in rows], dtype=float)
    return data[:, :8], data[:, 8].astype(int)


def load_heart(path):
    rows = _read_rows(path)
    data = np.array([[_num(v) for v in r] for r in rows], dtype=float)
    X, y = data[:, :13], data[:, 13]
    return X, (y > 0).astype(int)  # 0 = no disease, 1-4 = disease present


def load_liver(path):
    rows = _read_rows(path)
    X, y = [], []
    for r in rows:
        gender = 1.0 if r[1].strip().lower() == "male" else 0.0
        X.append([_num(r[0]), gender] + [_num(v) for v in r[2:10]])
        y.append(1 if r[10].strip() == "1" else 0)  # 1 = liver patient, 2 = not
    return np.array(X, dtype=float), np.array(y, dtype=int)


LOADERS = {"diabetes": load_diabetes, "heart": load_heart, "liver": load_liver}


# --------------------------------------------------------------------- pipelines
def build_preprocessor(disease: str) -> ColumnTransformer:
    spec = DISEASES[disease]
    feats = spec["features"]
    idx = {f: i for i, f in enumerate(feats)}
    zero_missing = spec.get("zero_as_missing", [])
    categorical = spec.get("categorical", [])
    log_cols = spec.get("log_transform", [])
    plain = [f for f in feats if f not in zero_missing + categorical + log_cols]

    transformers = []
    if zero_missing:
        transformers.append((
            "zero_missing",
            Pipeline([
                ("impute", SimpleImputer(missing_values=0, strategy="median")),
                ("scale", StandardScaler()),
            ]),
            [idx[f] for f in zero_missing],
        ))
    if log_cols:
        transformers.append((
            "log",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("log1p", FunctionTransformer(np.log1p)),
                ("scale", StandardScaler()),
            ]),
            [idx[f] for f in log_cols],
        ))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            [idx[f] for f in categorical],
        ))
    if plain:
        transformers.append((
            "numeric",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            [idx[f] for f in plain],
        ))
    return ColumnTransformer(transformers)


def candidates():
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


# ---------------------------------------------------------------------- training
def train_disease(disease: str) -> dict:
    X, y = LOADERS[disease](download(disease))
    logger.info("[%s] %d samples, %d features, positive rate %.2f", disease, X.shape[0], X.shape[1], y.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cv_scores = {}
    for name, clf in candidates().items():
        pipe = Pipeline([("prep", build_preprocessor(disease)), ("clf", clf)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
        cv_scores[name] = round(float(scores.mean()), 4)
        logger.info("[%s] %-20s CV F1 = %.4f", disease, name, scores.mean())

    best_name = max(cv_scores, key=cv_scores.get)
    best = Pipeline([("prep", build_preprocessor(disease)), ("clf", candidates()[best_name])])
    best.fit(X_train, y_train)
    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    test_metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    logger.info("[%s] selected %s -> test %s", disease, best_name, test_metrics)

    # Refit on all available data for the production model.
    final = Pipeline([("prep", build_preprocessor(disease)), ("clf", candidates()[best_name])])
    final.fit(X, y)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(final, os.path.join(MODELS_DIR, f"{disease}.joblib"), compress=3)

    return {
        "model": best_name,
        "dataset": DISEASES[disease]["dataset"],
        "n_samples": int(X.shape[0]),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "positive_rate": round(float(y.mean()), 4),
        "cv_f1_by_model": cv_scores,
        "test_metrics": test_metrics,
    }


def main():
    results = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "diseases": {d: train_disease(d) for d in DISEASES},
    }
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n================ EVALUATION SUMMARY (held-out 20% test set) ================")
    print(f"{'disease':<10}{'model':<22}{'accuracy':>9}{'precision':>10}{'recall':>8}{'f1':>8}{'roc_auc':>9}")
    for d, r in results["diseases"].items():
        m = r["test_metrics"]
        print(f"{d:<10}{r['model']:<22}{m['accuracy']:>9}{m['precision']:>10}{m['recall']:>8}{m['f1']:>8}{m['roc_auc']:>9}")
    print(f"\nModels + metrics saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
