import pandas as pd


def merge_and_clean(train: pd.DataFrame, weather: pd.DataFrame, buildings: pd.DataFrame) -> pd.DataFrame:
    """Merge the ASHRAE datasets and drop rows with missing values."""
    df = train.merge(buildings, on="building_id", how="left")
    df = df.merge(weather, on=["site_id", "timestamp"], how="left")
    df = df.dropna()
    return df
