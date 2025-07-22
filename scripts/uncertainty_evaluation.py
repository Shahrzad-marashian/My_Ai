import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib


def main() -> None:
    """Compute prediction uncertainty for RandomForest predictions."""
    tree_preds_path = Path("results/tree_predictions.npy")
    test_data_path = Path("results/test_data.pkl")
    model_path = Path("results/best_sample_model.pkl")
    output_csv = Path("results/predictions_with_uncertainty.csv")
    plot_path = Path("results/pred_vs_uncertainty.png")

    # Load test data (features, targets, and optional info)
    if not test_data_path.exists():
        raise FileNotFoundError(f"Missing test data at {test_data_path}")
    X_test, y_test, _ = joblib.load(test_data_path)

    # Get predictions from each tree in the forest
    if tree_preds_path.exists():
        all_tree_preds = np.load(tree_preds_path)
    else:
        if not model_path.exists():
            raise FileNotFoundError(f"Missing RandomForest model at {model_path}")
        model = joblib.load(model_path)
        X_filled = X_test.fillna(0)
        all_tree_preds = np.vstack([est.predict(X_filled) for est in model.estimators_])
        np.save(tree_preds_path, all_tree_preds)

    prediction_mean = all_tree_preds.mean(axis=0)
    prediction_std = all_tree_preds.std(axis=0)
    actual = np.expm1(y_test)

    result = pd.DataFrame({
        "id": np.arange(len(actual)),
        "prediction_mean": prediction_mean,
        "prediction_std": prediction_std,
        "actual_meter_reading": actual,
    })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)

    plt.figure()
    plt.scatter(prediction_mean, prediction_std, alpha=0.6)
    plt.xlabel("Prediction Mean")
    plt.ylabel("Prediction Std")
    plt.title("Prediction vs Uncertainty")
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":
    main()
