import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет строки с пропущенными значениями.

    Args:
        df: Исходный DataFrame.

    Returns:
        Очищенный DataFrame.
    """
    df = df.dropna()
    return df
