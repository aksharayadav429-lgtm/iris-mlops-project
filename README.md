# Iris MLOps Beginner Project

This project demonstrates the MLOps stages using the built-in Iris flower dataset.

## Workflow
1. Data collection and validation
2. Model training and experiment metrics
3. Evaluation and model artifact versioning
4. Deployment with FastAPI
5. CI/CD testing with GitHub Actions
6. Basic prediction logging for monitoring

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/train.py
```

The training script creates the `artifacts` folder containing:
- `iris_validated.csv`: validated dataset
- `metrics.json`: model parameters and evaluation metrics
- `iris_model.joblib`: trained model artifact

## Run the API

```powershell
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs to test the API.

Example request to `POST /predict`:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Predictions are appended to `artifacts/prediction_log.csv`.

## Tests and CI/CD

After training the model, run:

```powershell
pytest -q
```

Push the project to GitHub. The workflow in `.github/workflows/ci.yml` installs dependencies, trains the model, and runs tests on each push or pull request.

## Important notes
- Accuracy can vary with model choice and the train/test split. Do not expect 100% accuracy every time.
- The prediction log is a basic demonstration, not production-grade monitoring.
- Retraining should be based on validated new data and measured performance/data drift, not performed automatically just because time has passed.
- This demo uses a local saved model. Cloud deployment, Docker, MLflow experiment tracking, Prometheus, Evidently, and Grafana can be added as later extensions.
