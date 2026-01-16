# src/preprocessing.py
import pandas as pd
import numpy as np


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Выполняет предобработку данных о поездках Uber."""
    # 1. Фильтрация по стоимости и пассажирам
    df_filtered = df[
        (df["fare_amount"] > 0)
        & (df["passenger_count"] > 0)
        & (df["passenger_count"] <= 6)
    ]

    # 2. Создание нового признака 'distance'
    df_filtered["distance"] = np.sqrt(
        (df_filtered["dropoff_longitude"] - df_filtered["pickup_longitude"]) ** 2
        + (df_filtered["dropoff_latitude"] - df_filtered["pickup_latitude"]) ** 2
    )
    final_features = df_filtered[["distance", "passenger_count"]]
    return final_features
