import pandas as pd
from pathlib import Path


def preprocess_samples(data_dir: Path) -> pd.DataFrame:
    """Load sample CSVs, merge them, add time features, and return the DataFrame."""
    # Load the sample CSV files
    train = pd.read_csv(data_dir / "train_sample.csv")
    buildings = pd.read_csv(data_dir / "building_metadata_sample.csv")
    weather = pd.read_csv(data_dir / "weather_train_sample.csv")

    # Merge train with buildings on 'building_id'
    df = train.merge(buildings, on="building_id", how="left")

    # Merge with weather on ['site_id', 'timestamp']
    df = df.merge(weather, on=["site_id", "timestamp"], how="left")

    # Convert timestamp column to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Extract time based features
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    df["month"] = df["timestamp"].dt.month

    return df


def main():
    data_dir = Path("data/sample")
    df = preprocess_samples(data_dir)

    # Print DataFrame shape and first few rows
    print("Merged DataFrame shape:", df.shape)
    print(df.head())

    # Print count of missing values per column
    print("\nMissing values per column:")
    print(df.isna().sum())


if __name__ == "__main__":
    main()
