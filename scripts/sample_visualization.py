import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend for script execution
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sample_preprocess import preprocess_samples


def plot_sample_visualizations(df: pd.DataFrame, out_dir: Path) -> None:
    """Create plots for the preprocessed sample dataset and save them."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Add log1p transformed column
    df["log_meter_reading"] = np.log1p(df["meter_reading"])

    # 1. Histograms
    plt.figure()
    sns.histplot(df["meter_reading"], bins=50)
    plt.title("Meter Reading Distribution (Raw)")
    plt.xlabel("meter_reading")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_meter_reading.png")

    plt.figure()
    sns.histplot(df["log_meter_reading"], bins=50)
    plt.title("Meter Reading Distribution (Log-Transformed)")
    plt.xlabel("log1p(meter_reading)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_log_meter_reading.png")

    plt.figure()
    sns.histplot(df["square_feet"], bins=30)
    plt.title("Square Feet Distribution")
    plt.xlabel("square_feet")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_square_feet.png")

    plt.figure()
    sns.histplot(df["air_temperature"], bins=30)
    plt.title("Air Temperature Distribution")
    plt.xlabel("air_temperature")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_air_temperature.png")

    # 2. Average meter_reading by hour
    plt.figure()
    hourly_avg = df.groupby("hour")["meter_reading"].mean()
    sns.lineplot(x=hourly_avg.index, y=hourly_avg.values)
    plt.title("Average Meter Reading by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Average Meter Reading")
    plt.tight_layout()
    plt.savefig(out_dir / "meter_reading_by_hour.png")

    # 3. Correlation heatmap for numeric columns
    plt.figure()
    numeric_cols = df.select_dtypes(include="number")
    corr = numeric_cols.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_heatmap.png")

    # 4. Optional boxplot of meter_reading by primary_use
    if "primary_use" in df.columns:
        plt.figure()
        sns.boxplot(x="primary_use", y="meter_reading", data=df)
        plt.xticks(rotation=45, ha="right")
        plt.title("Meter Reading by Primary Use")
        plt.xlabel("Primary Use")
        plt.ylabel("Meter Reading")
        plt.tight_layout()
        plt.savefig(out_dir / "meter_reading_by_primary_use.png")

    plt.close("all")


def main() -> None:
    data_dir = Path("data/sample")
    df = preprocess_samples(data_dir)
    plot_dir = Path("results/sample_plots")
    plot_sample_visualizations(df, plot_dir)


if __name__ == "__main__":
    main()
