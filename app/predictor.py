"""Model loading and inference."""

import json
import logging
import os
from typing import Dict, Optional

import joblib
import numpy as np

from app.specs import DISCLAIMER, DISEASES

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "models"))

_models: Dict[str, object] = {}
_metrics: Optional[dict] = None


def load_models() -> None:
    """Load all trained pipelines and the metrics file into memory."""
    global _metrics
    for disease in DISEASES:
        path = os.path.join(MODELS_DIR, f"{disease}.joblib")
        if os.path.exists(path):
            try:
                _models[disease] = joblib.load(path)
                logger.info("Loaded model for %s from %s", disease, path)
            except Exception:
                logger.exception("Failed to load model for %s", disease)
        else:
            logger.warning("Model file missing for %s: %s (run train_models.py)", disease, path)
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            _metrics = json.load(f)


def is_loaded(disease: str) -> bool:
    return disease in _models


def get_metrics() -> Optional[dict]:
    return _metrics


def _risk_level(p: float) -> str:
    if p >= 0.6:
        return "high"
    if p >= 0.3:
        return "moderate"
    return "low"


def _to_vector(disease: str, payload: dict) -> np.ndarray:
    row = []
    for feat in DISEASES[disease]["features"]:
        v = payload.get(feat)
        if feat == "gender":
            v = 1.0 if v == "male" else 0.0
        row.append(np.nan if v is None else float(v))
    return np.array([row], dtype=float)


def predict(disease: str, payload: dict) -> dict:
    if disease not in _models:
        raise RuntimeError(f"Model for '{disease}' is not loaded")
    spec = DISEASES[disease]
    model = _models[disease]
    X = _to_vector(disease, payload)
    prob = float(model.predict_proba(X)[0, 1])
    has_disease = prob >= 0.5
    model_name = (_metrics or {}).get("diseases", {}).get(disease, {}).get("model") or type(model.named_steps["clf"]).__name__
    return {
        "disease": disease,
        "prediction": spec["positive_label"] if has_disease else spec["negative_label"],
        "has_disease": has_disease,
        "probability": round(prob, 4),
        "risk_level": _risk_level(prob),
        "model": model_name,
        "disclaimer": DISCLAIMER,
    }
