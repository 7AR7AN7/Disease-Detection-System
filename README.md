# Multi-Disease Detection API

FastAPI + scikit-learn REST API that estimates the risk of **diabetes**, **heart disease** and **liver disease** from tabular patient data.

> **Disclaimer:** For informational/educational purposes only. NOT medical advice or a diagnosis.

## Datasets
| Disease | Dataset | Samples |
|---|---|---|
| diabetes | Pima Indians Diabetes Database | 768 |
| heart | UCI Heart Disease (Cleveland) | 303 |
| liver | Indian Liver Patient Dataset (ILPD) | 583 |

## Training pipeline (`train_models.py`)
1. Downloads datasets into `data/` (cached).
2. Data prep: Pima zeros -> missing + median imputation; heart `?` -> missing, target binarised (0 vs 1-4), one-hot for categorical codes; liver gender encoding, log1p on skewed lab values, A/G ratio imputation; standard scaling.
3. Compares Logistic Regression, Random Forest and Gradient Boosting with 5-fold CV (F1) and selects the best.
4. Evaluates on a stratified 20% hold-out set (accuracy, precision, recall, F1, ROC-AUC, confusion matrix).
5. Refits on all data, saves `models/<disease>.joblib` and `models/metrics.json`.

```bash
.venv/bin/python3 -m pip install -r requirements.txt
.venv/bin/python3 train_models.py
```

### Current results (hold-out test set)
| Disease | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| diabetes | random_forest | 0.734 | 0.603 | 0.704 | 0.650 | 0.819 |
| heart | logistic_regression | 0.869 | 0.813 | 0.929 | 0.867 | 0.958 |
| liver | gradient_boosting | 0.692 | 0.737 | 0.880 | 0.802 | 0.751 |

## Run the API
```bash
PORT=3000 sh start.sh
```
Interactive docs: `http://localhost:3000/docs`

## Endpoints
| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service + model load status |
| GET | `/models` | Diseases, required input fields, evaluation metrics |
| POST | `/predict/diabetes` | Diabetes risk |
| POST | `/predict/heart` | Heart disease risk |
| POST | `/predict/liver` | Liver disease risk |

## Examples
```bash
curl -X POST http://localhost:3000/predict/diabetes -H 'Content-Type: application/json' -d '{
  "pregnancies": 2, "glucose": 148, "blood_pressure": 72, "skin_thickness": 35,
  "insulin": 0, "bmi": 33.6, "diabetes_pedigree_function": 0.627, "age": 50}'

curl -X POST http://localhost:3000/predict/heart -H 'Content-Type: application/json' -d '{
  "age": 63, "sex": 1, "chest_pain_type": 4, "resting_blood_pressure": 145, "cholesterol": 233,
  "fasting_blood_sugar": 1, "resting_ecg": 2, "max_heart_rate": 150, "exercise_angina": 0,
  "st_depression": 2.3, "st_slope": 3, "num_major_vessels": 0, "thalassemia": 6}'

curl -X POST http://localhost:3000/predict/liver -H 'Content-Type: application/json' -d '{
  "age": 62, "gender": "male", "total_bilirubin": 10.9, "direct_bilirubin": 5.5,
  "alkaline_phosphotase": 699, "alamine_aminotransferase": 64, "aspartate_aminotransferase": 100,
  "total_proteins": 7.5, "albumin": 3.2, "albumin_globulin_ratio": 0.74}'
```

Example response:
```json
{
  "disease": "diabetes",
  "prediction": "diabetes",
  "has_disease": true,
  "probability": 0.868,
  "risk_level": "high",
  "model": "random_forest",
  "disclaimer": "This prediction ... is NOT medical advice ..."
}
```
`risk_level`: low (<0.3), moderate (0.3-0.6), high (>=0.6). Invalid inputs return HTTP 422 with details. Optional fields (`num_major_vessels`, `thalassemia`, `albumin_globulin_ratio`) can be omitted/null and are imputed.

## Tests
```bash
.venv/bin/python3 -m pytest -q
```
