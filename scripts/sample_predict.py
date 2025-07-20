import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score


def generate_predictions() -> None:
    """Load the trained model and test data, make predictions, and save to CSV."""
    model_path = Path("results/best_sample_model.pkl")
    test_data_path = Path("results/test_data.pkl")
    output_csv = Path("results/sample_predictions.csv")

    # Load saved objects
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    print(f"Loading test data from {test_data_path}")
    X_test, y_test, info_test = joblib.load(test_data_path)

    print(f"Predicting on X_test with shape {X_test.shape}")
    preds_log = model.predict(X_test.fillna(0))

    rmse = np.sqrt(mean_squared_error(y_test, preds_log))
    r2 = r2_score(y_test, preds_log)
    print(f"Evaluation on log scale - RMSE: {rmse:.4f}, R^2: {r2:.4f}")

    preds = np.expm1(preds_log)
    actual = np.expm1(y_test)

    result = info_test.copy()
    result["meter_reading"] = preds
    result["actual_meter_reading"] = actual

    # Ensure timestamp is datetime and sort
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result = result.sort_values("timestamp")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")


if __name__ == "__main__":
    generate_predictions()
