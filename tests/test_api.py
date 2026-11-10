from fastapi.testclient import TestClient

from app.main import app

DIABETES = {
    "pregnancies": 2, "glucose": 148, "blood_pressure": 72, "skin_thickness": 35,
    "insulin": 0, "bmi": 33.6, "diabetes_pedigree_function": 0.627, "age": 50,
}
HEART = {
    "age": 63, "sex": 1, "chest_pain_type": 4, "resting_blood_pressure": 145, "cholesterol": 233,
    "fasting_blood_sugar": 1, "resting_ecg": 2, "max_heart_rate": 150, "exercise_angina": 0,
    "st_depression": 2.3, "st_slope": 3, "num_major_vessels": 0, "thalassemia": 6,
}
LIVER = {
    "age": 62, "gender": "male", "total_bilirubin": 10.9, "direct_bilirubin": 5.5,
    "alkaline_phosphotase": 699, "alamine_aminotransferase": 64, "aspartate_aminotransferase": 100,
    "total_proteins": 7.5, "albumin": 3.2, "albumin_globulin_ratio": 0.74,
}


def test_health_and_models():
    with TestClient(app) as c:
        h = c.get("/health").json()
        assert h["status"] == "healthy"
        m = c.get("/models").json()
        assert {x["disease"] for x in m["models"]} == {"diabetes", "heart", "liver"}
        for x in m["models"]:
            assert {"accuracy", "precision", "recall", "f1"} <= set(x["metrics"])


def test_predictions():
    with TestClient(app) as c:
        for disease, body in (("diabetes", DIABETES), ("heart", HEART), ("liver", LIVER)):
            r = c.post(f"/predict/{disease}", json=body)
            assert r.status_code == 200, r.text
            j = r.json()
            assert 0 <= j["probability"] <= 1
            assert "not medical advice" in j["disclaimer"].lower()


def test_optional_fields_and_validation():
    with TestClient(app) as c:
        body = {**HEART, "num_major_vessels": None, "thalassemia": None}
        assert c.post("/predict/heart", json=body).status_code == 200
        assert c.post("/predict/diabetes", json={**DIABETES, "age": -5}).status_code == 422
        assert c.post("/predict/liver", json={**LIVER, "gender": "x"}).status_code == 422
        assert c.post("/predict/heart", json={"age": 50}).status_code == 422
