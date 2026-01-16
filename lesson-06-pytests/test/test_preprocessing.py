import pytest
import numpy as np
import pandas as pd
from src.preprocessing import preprocess_data


def test_preprocess_does_not_modify_input_dataframe() -> None:
    original_data = pd.DataFrame(
        {
            "fare_amount": [10.0, 20.0],
            "passenger_count": [1, 2],
            "pickup_longitude": [-73.9, -73.9],
            "pickup_latitude": [40.7, 40.7],
            "dropoff_longitude": [-74.0, -74.0],
            "dropoff_latitude": [40.8, 40.8],
        }
    )

    original_columns = list(original_data.columns)
    preprocess_data(original_data)
    assert list(original_data.columns) == original_columns


def test_filtering_keeps_only_valid_rows(raw_uber_data: pd.DataFrame) -> None:
    result = preprocess_data(raw_uber_data)

    assert len(result) == 2


def test_distance_is_calculated_correctly(raw_uber_data: pd.DataFrame) -> None:
    result = preprocess_data(raw_uber_data)

    # Берём первую валидную строку
    distance = result["distance"].iloc[0]

    # Ожидаемое значение
    expected_distance = np.sqrt((-74.0 + 73.9) ** 2 + (40.8 - 40.7) ** 2)

    assert distance == pytest.approx(expected_distance)


def test_output_contains_only_expected_columns(raw_uber_data: pd.DataFrame) -> None:
    result = preprocess_data(raw_uber_data)

    assert list(result.columns) == ["distance", "passenger_count"]


@pytest.mark.parametrize(
    "fare_amount, passenger_count",
    [
        (0.0, 1),  # fare_amount = 0
        (-5.0, 2),  # fare_amount < 0
        (10.0, 0),  # passenger_count = 0
        (10.0, 7),  # passenger_count > 6
    ],
)
def test_filtering_edge_cases(
    fare_amount: float,
    passenger_count: int,
) -> None:
    df = pd.DataFrame(
        {
            "fare_amount": [10.0, fare_amount],
            "passenger_count": [1, passenger_count],
            "pickup_longitude": [-73.9, -73.9],
            "pickup_latitude": [40.7, 40.7],
            "dropoff_longitude": [-74.0, -74.0],
            "dropoff_latitude": [40.8, 40.8],
        }
    )

    result = preprocess_data(df)

    # Должна остаться только одна валидная строка
    assert len(result) == 1
