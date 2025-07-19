import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # ensure non-interactive backend for script execution
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from sample_preprocess import preprocess_samples
from sample_ml_prep import prepare_for_ml


def plot_model_performance(model, X_test: pd.DataFrame, y_test: pd.Series, out_dir: Path) -> None:
    """Generate diagnostic plots for a trained regression model."""
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = model.predict(X_test)
    residuals = y_test - preds

    # Predicted vs actual
    plt.figure()
    sns.scatterplot(x=y_test, y=preds, alpha=0.6)
    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(out_dir / "pred_vs_actual.png")

    # Residuals histogram
    plt.figure()
    sns.histplot(residuals, bins=30)
    plt.title("Residuals Histogram")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "residuals_hist.png")

    # Residuals vs predicted
    plt.figure()
    sns.scatterplot(x=preds, y=residuals, alpha=0.6)
    plt.axhline(0, color="r", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted")
    plt.tight_layout()
    plt.savefig(out_dir / "residuals_vs_pred.png")

    # Feature importances
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X_test.columns)
        top10 = importances.nlargest(10)
        plt.figure(figsize=(8, 6))
        sns.barplot(x=top10.values, y=top10.index)
        plt.title("Top 10 Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(out_dir / "feature_importances.png")

    plt.close("all")


def train_and_compare(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    save_path: Path | None = None,
    plot_dir: Path | None = None,
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

    # Select best model based on R^2
    best_name = max(metrics, key=lambda k: metrics[k][2])
    best_model = models[best_name]

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, save_path)
        print(f"Saved {best_name} model to {save_path}")

    if plot_dir is not None:
        plot_model_performance(best_model, X_test_filled, y_test, plot_dir)


def main() -> None:
    data_dir = Path("data/sample")
    df = preprocess_samples(data_dir)
    X_train, X_test, y_train, y_test = prepare_for_ml(df)
    save_path = Path("results/best_sample_model.pkl")
    plot_dir = Path("results/model_plots")
    train_and_compare(X_train, X_test, y_train, y_test, save_path, plot_dir)


if __name__ == "__main__":
    main()
