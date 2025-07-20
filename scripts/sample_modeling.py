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
from sklearn.model_selection import RandomizedSearchCV
import joblib

from sample_preprocess import preprocess_samples
from sample_ml_prep import prepare_for_ml
from sklearn.model_selection import train_test_split


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


def tune_random_forest(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> RandomForestRegressor:
    """Hyperparameter tuning for RandomForestRegressor using RandomizedSearchCV."""
    param_dist = {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, 40, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring="neg_mean_squared_error",
        random_state=42,
        n_jobs=-1,
    )

    X_train_filled = X_train.fillna(0)
    search.fit(X_train_filled, y_train)
    print("Best RandomForest Params:", search.best_params_)
    print(f"Best CV Score (neg MSE): {search.best_score_:.4f}")

    best_model = search.best_estimator_
    # Retrain on the full training data for evaluation consistency
    best_model.fit(X_train_filled, y_train)

    preds = best_model.predict(X_test.fillna(0))
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"Tuned RandomForest - RMSE: {rmse:.4f}, R^2: {r2:.4f}")
    return best_model


def main() -> None:
    data_dir = Path("data/sample")
    df = preprocess_samples(data_dir)

    # Preserve identifying columns for later predictions
    info = df[["building_id", "timestamp"]].copy()

    X_train, X_test, y_train, y_test = prepare_for_ml(df)

    # Split the info dataframe in the same way as features
    _, info_test = train_test_split(info, test_size=0.2, random_state=42)

    plot_dir = Path("results/model_plots")
    train_and_compare(X_train, X_test, y_train, y_test, None, plot_dir)

    # Hyperparameter tuning and evaluation
    best_model = tune_random_forest(X_train, X_test, y_train, y_test)

    # Save the best model
    model_path = Path("results/best_sample_model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"Saved best model to {model_path}")

    # Save the test data for the prediction script
    test_data_path = Path("results/test_data.pkl")
    joblib.dump((X_test, y_test, info_test), test_data_path)
    print(f"Saved test data to {test_data_path}")


if __name__ == "__main__":
    main()
