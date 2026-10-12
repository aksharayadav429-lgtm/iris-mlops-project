
from pathlib import Path
from datetime import datetime
import json
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

CURRENT_MODEL = ARTIFACTS / "iris_model.joblib"
METRICS_FILE = ARTIFACTS / "metrics.json"
CANDIDATE_MODEL = ARTIFACTS / "iris_model_candidate.joblib"

# Initial acceptance rule for this demo
MINIMUM_ACCURACY = 0.90


def main():
    print("Starting retraining workflow...")

    # 1. Load and split the Iris dataset
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 2. Train a candidate model
    candidate = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    candidate.fit(X_train, y_train)

    # 3. Evaluate the candidate
    predictions = candidate.predict(X_test)
    candidate_accuracy = float(accuracy_score(y_test, predictions))

    print(f"Candidate accuracy: {candidate_accuracy:.2%}")

    # 4. Read current recorded accuracy
    current_accuracy = None

    if METRICS_FILE.exists():
        with METRICS_FILE.open("r", encoding="utf-8") as file:
            old_metrics = json.load(file)
        current_accuracy = old_metrics.get("accuracy")

    if current_accuracy is not None:
        print(f"Previously recorded accuracy: {current_accuracy:.2%}")

    # 5. Apply the acceptance rule
    if candidate_accuracy < MINIMUM_ACCURACY:
        print(
            f"REJECTED: Accuracy is below "
            f"{MINIMUM_ACCURACY:.0%}."
        )
        return

    if (
        current_accuracy is not None
        and candidate_accuracy < current_accuracy
    ):
        print("REJECTED: Candidate is worse than the current recorded accuracy.")
        return

    # 6. Save candidate first, then promote it
    joblib.dump(
        {
            "model": candidate,
            "feature_names": list(X.columns),
            "target_names": list(iris.target_names),
        },
        CANDIDATE_MODEL,
    )

    # Verify that the candidate artifact can be loaded
    verified = joblib.load(CANDIDATE_MODEL)
    if not hasattr(verified["model"], "predict"):
        CANDIDATE_MODEL.unlink(missing_ok=True)
        print("REJECTED: Saved candidate failed verification.")
        return

    # Keep a backup of the current model before replacement
    backup_model = ARTIFACTS / "iris_model_previous.joblib"

    if CURRENT_MODEL.exists():
        joblib.copy if False else None
        import shutil
        shutil.copy2(CURRENT_MODEL, backup_model)

    CANDIDATE_MODEL.replace(CURRENT_MODEL)

    # Record the accepted model's evaluation
    new_metrics = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": "RandomForestClassifier",
        "accuracy": round(candidate_accuracy, 4),
        "minimum_accuracy": MINIMUM_ACCURACY,
        "retraining_status": "accepted",
        "random_state": 42,
        "test_size": 0.2,
    }

    METRICS_FILE.write_text(
        json.dumps(new_metrics, indent=2),
        encoding="utf-8",
    )

    print("RETRAINING ACCEPTED")
    print(f"Updated model: {CURRENT_MODEL}")
    print(f"Metrics saved: {METRICS_FILE}")

    if backup_model.exists():
        print(f"Previous model backup: {backup_model}")


if __name__ == "__main__":
    main()
