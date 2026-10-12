from pathlib import Path
from datetime import datetime
import csv
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "iris_model.joblib"
LOG_PATH = ROOT / "artifacts" / "prediction_log.csv"

app = FastAPI(title="Iris MLOps Demo API", version="1.0.0")
bundle = None

class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

@app.on_event("startup")
def load_model():
    global bundle
    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return {"message": "Iris MLOps API is running", "model_loaded": bundle is not None}

@app.post("/predict")
def predict(data: IrisInput):
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not found. Run python src/train.py first.")
    values = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    model = bundle["model"]
    predicted_id = int(model.predict(values)[0])
    species = bundle["target_names"][predicted_id]
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = round(float(max(model.predict_proba(values)[0])), 4)

    LOG_PATH.parent.mkdir(exist_ok=True)
    new_file = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "sepal_length", "sepal_width", "petal_length", "petal_width", "prediction", "confidence"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), *values[0], species, confidence])

    return {"predicted_species": species, "confidence": confidence}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bundle is not None}
