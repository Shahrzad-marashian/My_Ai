import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sample_preprocess import preprocess_samples


def prepare_for_ml(df: pd.DataFrame):
    """Prepare the merged sample dataframe for machine learning."""
    # Step 1: drop unneeded columns
    df = df.drop(columns=["building_id", "timestamp", "site_id"], errors="ignore")

    # Step 2: create log_meter_reading as target
    df["log_meter_reading"] = np.log1p(df["meter_reading"])
    df = df.drop(columns=["meter_reading"])  # drop original target

    # Step 3: one-hot encode the primary_use column
    df = pd.get_dummies(df, columns=["primary_use"], drop_first=True)

    # Step 4: scale numerical features
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols.remove("log_meter_reading")
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Step 5: define features and target
    X = df.drop(columns=["log_meter_reading"])
    y = df["log_meter_reading"]

    # Step 6: train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 7: print shapes
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    return X_train, X_test, y_train, y_test


def main() -> None:
    data_dir = Path("data/sample")
    df = preprocess_samples(data_dir)
    prepare_for_ml(df)


if __name__ == "__main__":
    main()
