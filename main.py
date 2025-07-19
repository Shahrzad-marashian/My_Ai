from pathlib import Path
import pandas as pd

from scripts.preprocessing import merge_and_clean
from scripts.visualization import plot_distributions
from scripts.modeling import train_and_evaluate


def main():
    data_dir = Path("data/ashrae-energy-prediction")
    train = pd.read_csv(data_dir / "train.csv")
    weather = pd.read_csv(data_dir / "weather_train.csv")
    buildings = pd.read_csv(data_dir / "building_metadata.csv")

    df = merge_and_clean(train, weather, buildings)
    plot_distributions(df)
    model, rmsle = train_and_evaluate(df)
    print(f"RMSLE: {rmsle:.4f}")


if __name__ == "__main__":
    main()
