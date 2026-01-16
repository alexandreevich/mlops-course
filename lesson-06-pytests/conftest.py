# conftest.py
import pytest
import pandas as pd


@pytest.fixture
def raw_uber_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fare_amount": [10.0, 20.0, 0.0, -5.0, 15.0, 12.0],
            "passenger_count": [1, 2, 1, 2, 0, 7],
            "pickup_longitude": [-73.9] * 6,
            "pickup_latitude": [40.7] * 6,
            "dropoff_longitude": [-74.0] * 6,
            "dropoff_latitude": [40.8] * 6,
        }
    )
