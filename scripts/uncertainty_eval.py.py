import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def evaluate_uncertainty() -> None:
    """Compute prediction uncertainty from RandomForest estimators."""
    model_path = Path("results/best_sample_model.pkl")
    test_data_path = Path("results/test_data.pkl")
    output_csv = Path("results/sample_predictions_with_uncertainty.csv")
    plot_path = Path("results/model_plots/pred_vs_uncertainty.png")

    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    print(f"Loading test data from {test_data_path}")
    X_test, y_test, info_test = joblib.load(test_data_path)

    X_test_filled = X_test.fillna(0)
    print(f"Computing predictions from {len(model.estimators_)} trees...")
    all_preds_log = np.vstack([
        est.predict(X_test_filled) for est in model.estimators_
    ])

    preds_log = all_preds_log.mean(axis=0)
    std_log = all_preds_log.std(axis=0)
    lower_log = np.percentile(all_preds_log, 5, axis=0)
    upper_log = np.percentile(all_preds_log, 95, axis=0)

    preds = np.expm1(preds_log)
    std = np.expm1(preds_log + std_log) - np.expm1(preds_log)
    lower = np.expm1(lower_log)
    upper = np.expm1(upper_log)
    actual = np.expm1(y_test)

    result = info_test.copy()
    result["prediction"] = preds
    result["uncertainty"] = std
    result["lower_bound"] = lower
    result["upper_bound"] = upper
    result["actual"] = actual

    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result = result.sort_values("timestamp")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    print(f"Saved predictions with uncertainty to {output_csv}")

    # Scatter plot of predicted mean vs uncertainty
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.scatter(preds, std, alpha=0.6)
    plt.xlabel("Predicted Meter Reading")
    plt.ylabel("Uncertainty (std dev)")
    plt.title("Prediction vs Uncertainty")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    evaluate_uncertainty()