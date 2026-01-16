import pandas as pd


def add_wind_humidity_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет признак wind_humidity_ratio = wind_speed / humidity.

    Args:
        df: DataFrame с исходными данными.

    Returns:
        DataFrame с новым признаком.
    """
    df["wind_humidity_ratio"] = df["wind_speed"] / df["humidity"].replace(0, 0.1)
    return df
