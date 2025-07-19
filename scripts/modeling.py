import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
import numpy as np


def train_and_evaluate(df: pd.DataFrame):
    """Train RandomForest and evaluate using RMSLE."""
    features = ["square_feet", "air_temperature", "dew_temperature"]
    X = df[features]
    y = df["meter_reading"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmsle = np.sqrt(mean_squared_log_error(y_test, np.maximum(0, preds)))
    return model, rmsle
