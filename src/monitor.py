
from pathlib import Path
import pandas as pd

# Find the project and prediction log
ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "artifacts" / "prediction_log.csv"


def monitor_predictions():
    if not LOG_PATH.exists():
        print("No prediction log found. Make a prediction using the API first.")
        return

    df = pd.read_csv(LOG_PATH)

    if df.empty:
        print("The prediction log is empty.")
        return

    print("\n===== IRIS MODEL MONITORING REPORT =====")

    # Total predictions
    print(f"\nTotal predictions: {len(df)}")

    # Predictions for each species
    print("\nPredictions by species:")
    print(df["prediction"].value_counts().to_string())

    # Average confidence
    confidence = pd.to_numeric(df["confidence"], errors="coerce")
    valid_confidence = confidence.dropna()

    if not valid_confidence.empty:
        print(
            f"\nAverage confidence: "
            f"{valid_confidence.mean() * 100:.2f}%"
        )
    else:
        print("\nNo valid confidence values found.")

    # Check for missing values
    print("\nMissing values by column:")
    print(df.isnull().sum().to_string())

    # Check confidence values
    invalid_confidence = confidence[
        (confidence < 0) | (confidence > 1)
    ]

    print(f"\nInvalid confidence values: {len(invalid_confidence)}")

    print("\n===== MONITORING COMPLETE =====")


if __name__ == "__main__":
    monitor_predictions()
