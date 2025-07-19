import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from sample_preprocess import preprocess_samples
from sample_ml_prep import prepare_for_ml


def train_and_compare(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    save_path: Path | None = None,
) -> None:
    """Train two regressors and print evaluation metrics."""
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
    }
    metrics = {}

    # Fill missing values so models do not fail
    X_train_filled = X_train.fillna(0)
    X_test_filled = X_test.fillna(0)

    for name, model in models.items():
        model.fit(X_train_filled, y_train)
        preds = model.predict(X_test_filled)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        metrics[name] = (mse, rmse, r2)
        print(f"{name} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R^2: {r2:.4f}")

    if save_path is not None:
        best_name = max(metrics, key=lambda k: metrics[k][2])  # choose by R^2
        best_model = models[best_name]
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, save_path)
        print(f"Saved {best_name} model to {save_path}")


def main() -> None:
    data_dir = Path("data/sample")
    df = preprocess_samples(data_dir)
    X_train, X_test, y_train, y_test = prepare_for_ml(df)
    save_path = Path("results/best_sample_model.pkl")
    train_and_compare(X_train, X_test, y_train, y_test, save_path)


if __name__ == "__main__":
    main()
