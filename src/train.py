from pathlib import Path
from datetime import datetime
import json
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

def main():
    # 1. Data collection
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    df = X.copy()
    df["species"] = y

    # 2. Data validation
    assert len(df) == 150, "Unexpected number of Iris records"
    assert not df.isnull().any().any(), "Dataset contains missing values"
    assert X.shape[1] == 4, "Expected four input features"
    assert y.nunique() == 3, "Expected three species classes"

    # Save a copy of the validated data
    df.to_csv(ARTIFACTS / "iris_validated.csv", index=False)

    # 3. Training (fixed random_state makes the split reproducible)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluation
    predictions = model.predict(X_test)
    metrics = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": "RandomForestClassifier",
        "n_estimators": 100,
        "test_size": 0.2,
        "random_state": 42,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision_macro": round(float(precision_score(y_test, predictions, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test, predictions, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, target_names=iris.target_names, output_dict=True, zero_division=0
        ),
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # 5. Model versioning: save a named model artifact and metadata
    joblib.dump(
        {"model": model, "feature_names": list(X.columns), "target_names": list(iris.target_names)},
        ARTIFACTS / "iris_model.joblib"
    )
    print("Training complete.")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Metrics saved to: {ARTIFACTS / 'metrics.json'}")
    print(f"Model saved to: {ARTIFACTS / 'iris_model.joblib'}")
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(metrics["confusion_matrix"])

if __name__ == "__main__":
    main()
