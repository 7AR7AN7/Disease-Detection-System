"""Pydantic request/response schemas with clinical-range validation."""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class DiabetesInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "pregnancies": 2,
                "glucose": 148,
                "blood_pressure": 72,
                "skin_thickness": 35,
                "insulin": 0,
                "bmi": 33.6,
                "diabetes_pedigree_function": 0.627,
                "age": 50,
            }
        },
    )

    pregnancies: int = Field(..., ge=0, le=20, description="Number of times pregnant (0 for males)")
    glucose: float = Field(..., ge=0, le=300, description="Plasma glucose, 2h oral glucose tolerance test (mg/dL). 0 = unknown")
    blood_pressure: float = Field(..., ge=0, le=200, description="Diastolic blood pressure (mm Hg). 0 = unknown")
    skin_thickness: float = Field(..., ge=0, le=100, description="Triceps skin fold thickness (mm). 0 = unknown")
    insulin: float = Field(..., ge=0, le=900, description="2-hour serum insulin (mu U/ml). 0 = unknown")
    bmi: float = Field(..., ge=0, le=80, description="Body mass index (kg/m^2). 0 = unknown")
    diabetes_pedigree_function: float = Field(..., ge=0, le=3, description="Diabetes pedigree function (family history score)")
    age: int = Field(..., ge=1, le=120, description="Age in years")


class HeartInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "age": 63,
                "sex": 1,
                "chest_pain_type": 4,
                "resting_blood_pressure": 145,
                "cholesterol": 233,
                "fasting_blood_sugar": 1,
                "resting_ecg": 2,
                "max_heart_rate": 150,
                "exercise_angina": 0,
                "st_depression": 2.3,
                "st_slope": 3,
                "num_major_vessels": 0,
                "thalassemia": 6,
            }
        },
    )

    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    chest_pain_type: Literal[1, 2, 3, 4] = Field(
        ..., description="1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic"
    )
    resting_blood_pressure: float = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    cholesterol: float = Field(..., ge=80, le=700, description="Serum cholesterol (mg/dL)")
    fasting_blood_sugar: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dL (1 = true, 0 = false)")
    resting_ecg: Literal[0, 1, 2] = Field(
        ..., description="0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy"
    )
    max_heart_rate: float = Field(..., ge=50, le=250, description="Maximum heart rate achieved (bpm)")
    exercise_angina: Literal[0, 1] = Field(..., description="Exercise-induced angina (1 = yes, 0 = no)")
    st_depression: float = Field(..., ge=0, le=10, description="ST depression induced by exercise relative to rest")
    st_slope: Literal[1, 2, 3] = Field(..., description="Slope of peak exercise ST segment: 1 = upsloping, 2 = flat, 3 = downsloping")
    num_major_vessels: Optional[Literal[0, 1, 2, 3]] = Field(
        None, description="Number of major vessels (0-3) colored by fluoroscopy. Optional (null = unknown)"
    )
    thalassemia: Optional[Literal[3, 6, 7]] = Field(
        None, description="3 = normal, 6 = fixed defect, 7 = reversible defect. Optional (null = unknown)"
    )


class LiverInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "age": 62,
                "gender": "male",
                "total_bilirubin": 10.9,
                "direct_bilirubin": 5.5,
                "alkaline_phosphotase": 699,
                "alamine_aminotransferase": 64,
                "aspartate_aminotransferase": 100,
                "total_proteins": 7.5,
                "albumin": 3.2,
                "albumin_globulin_ratio": 0.74,
            }
        },
    )

    age: int = Field(..., ge=1, le=120, description="Age in years")
    gender: Literal["male", "female"] = Field(..., description="'male' or 'female'")
    total_bilirubin: float = Field(..., ge=0, le=100, description="Total bilirubin (mg/dL)")
    direct_bilirubin: float = Field(..., ge=0, le=50, description="Direct bilirubin (mg/dL)")
    alkaline_phosphotase: float = Field(..., ge=0, le=3000, description="Alkaline phosphatase (IU/L)")
    alamine_aminotransferase: float = Field(..., ge=0, le=3000, description="Alanine aminotransferase / ALT (IU/L)")
    aspartate_aminotransferase: float = Field(..., ge=0, le=5000, description="Aspartate aminotransferase / AST (IU/L)")
    total_proteins: float = Field(..., ge=0, le=15, description="Total proteins (g/dL)")
    albumin: float = Field(..., ge=0, le=10, description="Albumin (g/dL)")
    albumin_globulin_ratio: Optional[float] = Field(
        None, ge=0, le=5, description="Albumin/globulin ratio. Optional (null = unknown)"
    )


class PredictionResponse(BaseModel):
    disease: str = Field(..., description="Disease key: diabetes | heart | liver")
    prediction: str = Field(..., description="Predicted label, e.g. 'diabetes' or 'no_diabetes'")
    has_disease: bool = Field(..., description="True if the model predicts the disease is present")
    probability: float = Field(..., description="Model probability (0-1) that the disease is present")
    risk_level: Literal["low", "moderate", "high"] = Field(
        ..., description="low (<0.3), moderate (0.3-0.6), high (>=0.6) based on probability"
    )
    model: str = Field(..., description="Classifier used")
    disclaimer: str


class FieldInfo(BaseModel):
    name: str
    type: str
    required: bool
    description: Optional[str] = None
    allowed_values: Optional[List] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None


class ModelInfo(BaseModel):
    disease: str
    display_name: str
    endpoint: str
    loaded: bool
    dataset: str
    model: Optional[str] = None
    input_fields: List[FieldInfo]
    metrics: Optional[Dict] = None


class ModelsResponse(BaseModel):
    trained_at: Optional[str] = None
    models: List[ModelInfo]
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
