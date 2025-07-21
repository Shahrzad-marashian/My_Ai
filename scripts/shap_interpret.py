"""Generate SHAP plots for the trained model on the test set."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    model_path = Path("results/best_sample_model.pkl")
    test_data_path = Path("results/test_data.pkl")
    plot_dir = Path("results/shap_plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    print(f"Loading test data from {test_data_path}")
    X_test, y_test, info_test = joblib.load(test_data_path)

    X_sample = X_test.fillna(0).head(5000)

    print("Computing SHAP values on sample of size", len(X_sample))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Summary plot
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(plot_dir / "shap_summary.png")
    plt.close()

    mean_abs = np.abs(shap_values).mean(0)
    top_indices = np.argsort(mean_abs)[::-1][:3]
    for idx in top_indices:
        feat = X_sample.columns[idx]
        print(f"Creating dependence plot for {feat}")
        shap.dependence_plot(feat, shap_values, X_sample, show=False)
        plt.tight_layout()
        fname = plot_dir / f"shap_dependence_{feat}.png"
        plt.savefig(fname)
        plt.close()


if __name__ == "__main__":
    main()
