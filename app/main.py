import logging
import os
from contextlib import asynccontextmanager
from typing import Type

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app import predictor
from app.schemas import (
    DiabetesInput,
    FieldInfo,
    HealthResponse,
    HeartInput,
    LiverInput,
    ModelInfo,
    ModelsResponse,
    PredictionResponse,
)
from app.specs import DISCLAIMER, DISEASES

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("disease_api")

INPUT_SCHEMAS = {"diabetes": DiabetesInput, "heart": HeartInput, "liver": LiverInput}


@asynccontextmanager
async def lifespan(_: FastAPI):
    predictor.load_models()
    yield


app = FastAPI(
    title="Multi-Disease Detection API",
    description=(
        "Machine-learning REST API that estimates the risk of **diabetes**, **heart disease** and "
        "**liver disease** from tabular patient data. Models are scikit-learn pipelines trained on "
        "public research datasets (Pima Indians Diabetes, UCI Cleveland Heart Disease, Indian Liver "
        "Patient Dataset).\n\n**Disclaimer:** " + DISCLAIMER
    ),
    version="1.0.0",
    docs_url="/docs",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")


def _field_infos(schema: Type[BaseModel]) -> list:
    props = schema.model_json_schema()
    required = set(props.get("required", []))
    out = []
    for name, p in props["properties"].items():
        variants = p.get("anyOf", [p])
        main = next((v for v in variants if v.get("type") != "null"), p)
        out.append(
            FieldInfo(
                name=name,
                type=main.get("type", "string"),
                required=name in required,
                description=p.get("description"),
                allowed_values=main.get("enum") or ([main["const"]] if "const" in main else None),
                minimum=main.get("minimum"),
                maximum=main.get("maximum"),
            )
        )
    return out


def _run_prediction(disease: str, data: BaseModel) -> dict:
    if not predictor.is_loaded(disease):
        raise HTTPException(status_code=503, detail=f"Model for '{disease}' is not available. Run train_models.py.")
    try:
        result = predictor.predict(disease, data.model_dump())
    except Exception as exc:
        logger.exception("Prediction failed for %s", disease)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
    logger.info("Prediction %s -> %s (p=%.3f)", disease, result["prediction"], result["probability"])
    return result


@app.get("/", include_in_schema=False)
def root():
    return {"service": "Multi-Disease Detection API", "docs": "/docs", "models": "/models"}


@app.get("/health", response_model=HealthResponse, summary="Service health and model load status", tags=["System"])
def health():
    loaded = {d: predictor.is_loaded(d) for d in DISEASES}
    return {"status": "healthy" if all(loaded.values()) else "degraded", "models_loaded": loaded}


@app.get(
    "/models",
    response_model=ModelsResponse,
    summary="List supported diseases, required input fields and evaluation metrics",
    tags=["Models"],
)
def list_models():
    metrics = predictor.get_metrics() or {}
    disease_metrics = metrics.get("diseases", {})
    models = []
    for disease, spec in DISEASES.items():
        m = disease_metrics.get(disease, {})
        models.append(
            ModelInfo(
                disease=disease,
                display_name=spec["display_name"],
                endpoint=f"/predict/{disease}",
                loaded=predictor.is_loaded(disease),
                dataset=spec["dataset"],
                model=m.get("model"),
                input_fields=_field_infos(INPUT_SCHEMAS[disease]),
                metrics=m.get("test_metrics"),
            )
        )
    return ModelsResponse(trained_at=metrics.get("trained_at"), models=models, disclaimer=DISCLAIMER)


@app.post("/predict/diabetes", response_model=PredictionResponse, summary="Predict diabetes risk", tags=["Predictions"])
def predict_diabetes(data: DiabetesInput):
    """Estimate Type 2 diabetes risk. Use 0 for glucose/blood_pressure/skin_thickness/insulin/bmi if unknown."""
    return _run_prediction("diabetes", data)


@app.post("/predict/heart", response_model=PredictionResponse, summary="Predict heart disease risk", tags=["Predictions"])
def predict_heart(data: HeartInput):
    """Estimate heart disease risk from clinical and exercise-test data (UCI Cleveland encoding)."""
    return _run_prediction("heart", data)


@app.post("/predict/liver", response_model=PredictionResponse, summary="Predict liver disease risk", tags=["Predictions"])
def predict_liver(data: LiverInput):
    """Estimate liver disease risk from liver function test results."""
    return _run_prediction("liver", data)


@app.get("/version.json", include_in_schema=False)
def version():
    # Required deployment contract: serves static/version.json with CORS enabled.
    path = os.path.join(_STATIC_DIR, "version.json")
    if os.path.exists(path):
        return FileResponse(path, media_type="application/json", headers={"Access-Control-Allow-Origin": "*"})
    return JSONResponse({}, status_code=404, headers={"Access-Control-Allow-Origin": "*"})
