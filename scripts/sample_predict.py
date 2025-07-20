import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

from sample_preprocess import preprocess_samples
from sample_ml_prep import prepare_for_ml


def generate_predictions() -> None:
    """Load model, predict on test set, and save predictions to CSV."""
    data_dir = Path("data/sample")
    df = preprocess_samples(data_dir)

    # Keep info columns before ML preparation
    info = df[["building_id", "timestamp"]]

    # Prepare features and train/test split
    _, X_test, _, _ = prepare_for_ml(df.copy())

    # Split the info dataframe using the same random state
    _, info_test = train_test_split(info, test_size=0.2, random_state=42)

    model_path = Path("results/best_sample_model.pkl")
    model = joblib.load(model_path)

    preds_log = model.predict(X_test.fillna(0))
    preds = np.expm1(preds_log)

    output = info_test.copy()
    output["meter_reading"] = preds
    output.to_csv("results/sample_predictions.csv", index=False)
    print("Saved predictions to results/sample_predictions.csv")


if __name__ == "__main__":
    generate_predictions()
