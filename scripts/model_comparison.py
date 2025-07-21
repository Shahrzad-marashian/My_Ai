import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_summary_table(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Return a DataFrame summarizing model metrics.

    Parameters
    ----------
    metrics : dict
        Mapping of model name to a dictionary with keys 'R2', 'RMSE', and 'MAE'.
    """
    return pd.DataFrame(metrics).T[['R2', 'RMSE', 'MAE']]


def plot_model_comparison(summary: pd.DataFrame, out_path: Path) -> None:
    """Generate a horizontal bar chart comparing models across metrics."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
    for ax, metric in zip(axes, ['R2', 'RMSE', 'MAE']):
        values = summary[metric]
        ax.barh(summary.index, values, color='skyblue')
        ax.set_title(metric)
        ax.set_xlabel(metric)
        ax.set_xlim(left=0)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def main() -> None:
    """Create a comparison table and plot using example metrics."""
    metrics = {
        'Linear Regression': {'R2': 0.72, 'RMSE': 0.55, 'MAE': 0.42},
        'Random Forest': {'R2': 0.85, 'RMSE': 0.35, 'MAE': 0.28},
    }

    summary = create_summary_table(metrics)
    print("\nModel Performance Summary:\n")
    print(summary.to_markdown())

    plot_model_comparison(summary, Path('results/model_comparison.png'))


if __name__ == '__main__':
    main()
