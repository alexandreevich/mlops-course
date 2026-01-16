import pandas as pd


def load_data(data_path: str) -> pd.DataFrame:
    """
    Загружает данные из CSV-файла.

    Args:
        data_path: Путь к файлу с данными.

    Returns:
        DataFrame с данными.
    """
    df = pd.read_csv(data_path)
    return df
