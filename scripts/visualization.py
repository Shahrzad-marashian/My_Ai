import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_distributions(df: pd.DataFrame) -> None:
    """Plot histogram of meter readings."""
    sns.histplot(df["meter_reading"], bins=50, kde=False)
    plt.xlabel("Meter Reading")
    plt.ylabel("Count")
    plt.title("Distribution of Meter Readings")
    plt.show()
