from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]

def test_model_artifact_exists():
    assert (ROOT / "artifacts" / "iris_model.joblib").exists()

def test_model_predicts_valid_class():
    bundle = joblib.load(ROOT / "artifacts" / "iris_model.joblib")
    model = bundle["model"]
    prediction = int(model.predict([[5.1, 3.5, 1.4, 0.2]])[0])
    assert prediction in [0, 1, 2]
